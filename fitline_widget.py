"""Fitline — embeddable three-pane OLS workbench (Qt).

Drop this widget into any Qt main window:

    from fitline_widget import RegressionWorkbench

    window = QMainWindow()
    window.setCentralWidget(RegressionWorkbench())
    window.resize(1280, 800)
    window.show()

Layout
------
A header plus three resizable panes (QSplitter):

    ┌─────────────────────────────────────────────────────────┐
    │ Fitline                          [Split] [Overlay]      │
    ├────────────────┬────────────────────────────────────────┤
    │ PARAMETERS     │  MAIN CHART                            │
    │ dataset, degree│  scatter + fitted curve                │
    │ intercept, SE  │  CI / PI bands, residual stems         │
    │ overlay toggles│  residual histogram inset              │
    │                ├────────────────────────────────────────┤
    │                │  REGRESSION RESULTS                    │
    │                │  R², coefficients, residual diagnostics│
    └────────────────┴────────────────────────────────────────┘

Drag the splitter handles to resize. Overlay mode floats the
results card on top of the chart. Chart mode collapses the
parameter rail.

Requires: numpy, matplotlib, and PySide6 (or PyQt6).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QColor, QFont, QPalette
    from PySide6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover
    from PyQt6.QtCore import Qt
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtGui import QColor, QFont, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QFrame,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSlider,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

BG = "#0c0c0e"
SURFACE = "#141416"
FG = "#ecece8"
MUTED = "#8c8d91"
BORDER = "#2a2a2e"
FIT = "#d8dce3"
POINT = "#ecece8"
BAND = "#c5c9d1"
RESIDUAL = "#b08968"


# ---------------------------------------------------------------------------
# Data + OLS
# ---------------------------------------------------------------------------

def _rng(seed: int):
    rs = np.random.RandomState(seed)
    return rs


def _henry_hub():
    rs = _rng(20260413)
    i = np.arange(156)
    storage = 1200 + 1600 * ((np.sin((i / 156) * np.pi * 2) + 1) / 2) + 90 * rs.randn(156)
    price = np.clip(5.8 - 0.00155 * storage + 0.55 * rs.randn(156), 1.4, None)
    return storage, price, "Working gas in storage (Bcf)", "Henry Hub ($/MMBtu)"


def _equity_beta():
    rs = _rng(88421)
    mkt = rs.randn(140) * 1.8
    ret = 0.04 + 1.12 * mkt + rs.randn(140) * 1.05
    return mkt, ret, "Market excess return (%)", "Stock excess return (%)"


def _ad_spend():
    rs = _rng(33107)
    i = np.arange(80)
    spend = np.clip(8 + i * 1.15 + 2.2 * rs.randn(80), 5, None)
    sales = np.clip(42 + 3.4 * spend - 0.014 * spend ** 2 + 6.5 * rs.randn(80), 20, None)
    return spend, sales, "Advertising spend ($k)", "Weekly sales ($k)"


def _quadratic():
    rs = _rng(17)
    i = np.arange(90)
    x = -8 + i * 0.2 + 0.15 * rs.randn(90)
    y = 2.2 + 0.55 * x - 0.12 * x ** 2 + 1.35 * rs.randn(90)
    return x, y, "x", "y"


DATASETS = {
    "Henry Hub vs. working gas": _henry_hub,
    "Stock vs. market returns": _equity_beta,
    "Ad spend vs. sales": _ad_spend,
    "Synthetic quadratic": _quadratic,
}


@dataclass
class FitResult:
    n: int
    k: int
    df: int
    r2: float
    adj_r2: float
    rmse: float
    mae: float
    f_stat: float
    dw: float
    x_mean: float
    x_sd: float
    names: list[str]
    beta: np.ndarray
    se: np.ndarray
    t: np.ndarray
    x_grid: np.ndarray
    y_hat: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    pi_lo: np.ndarray
    pi_hi: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    equation: str


def _design(x: np.ndarray, mean: float, sd: float, degree: int, intercept: bool) -> np.ndarray:
    z = (x - mean) / sd
    cols = [np.ones_like(z)] if intercept else []
    power = np.ones_like(z)
    for _ in range(degree):
        power = power * z
        cols.append(power)
    return np.column_stack(cols)


def fit_ols(x: np.ndarray, y: np.ndarray, degree: int, intercept: bool, robust: bool) -> FitResult | None:
    n = len(x)
    k = int(intercept) + degree
    if n <= k:
        return None
    x_mean = float(x.mean())
    x_sd = float(max(x.std(), 1e-12))
    X = _design(x, x_mean, x_sd, degree, intercept)
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    sse = float(resid @ resid)
    df = n - k
    sigma2 = sse / df
    if robust:
        meat = (X * (resid ** 2)[:, None]).T @ X
        cov = (n / df) * (xtx_inv @ meat @ xtx_inv)
    else:
        cov = sigma2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.zeros_like(beta), where=se > 1e-12)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 1e-12 else 0.0
    adj = 1 - (1 - r2) * ((n - 1) / df)
    rmse = float(np.sqrt(sigma2))
    mae = float(np.mean(np.abs(resid)))
    msr = (sst - sse) / max(k - int(intercept), 1)
    f_stat = msr / sigma2 if sigma2 > 1e-12 else 0.0
    dw = float(np.sum(np.diff(resid) ** 2) / sse) if sse > 1e-12 else 0.0

    tcrit = 1.96 if df > 30 else 2.05
    span = float(x.max() - x.min()) or 1.0
    grid = np.linspace(x.min() - 0.04 * span, x.max() + 0.04 * span, 120)
    Xg = _design(grid, x_mean, x_sd, degree, intercept)
    yhat = Xg @ beta
    se_fit = np.sqrt(np.clip(np.sum(Xg * (Xg @ xtx_inv), axis=1) * sigma2, 0, None))
    se_pred = np.sqrt(sigma2 + se_fit ** 2)

    names = (["Intercept"] if intercept else []) + (["z"] + [f"z^{d}" for d in range(2, degree + 1)])
    parts = []
    for i, (name, b) in enumerate(zip(names, beta)):
        mag = abs(float(b))
        if i == 0 and name == "Intercept":
            parts.append(f"{float(b):.3f}")
        else:
            sign = "−" if b < 0 else "+"
            label = name.replace("^2", "²").replace("^3", "³")
            if i == 0:
                parts.append(f"{float(b):.3f} {label}")
            else:
                parts.append(f"{sign} {mag:.3f} {label}")
    equation = "ŷ = " + " ".join(parts)

    return FitResult(
        n=n, k=k, df=df, r2=r2, adj_r2=adj, rmse=rmse, mae=mae,
        f_stat=f_stat, dw=dw, x_mean=x_mean, x_sd=x_sd, names=names,
        beta=beta, se=se, t=t, x_grid=grid, y_hat=yhat,
        ci_lo=yhat - tcrit * se_fit, ci_hi=yhat + tcrit * se_fit,
        pi_lo=yhat - tcrit * se_pred, pi_hi=yhat + tcrit * se_pred,
        fitted=fitted, resid=resid, equation=equation,
    )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class RegressionWorkbench(QWidget):
    """Self-contained three-pane OLS workbench. Set as a QMainWindow central widget."""

    specChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout_mode = "split"
        self._apply_palette()
        self._x = np.array([])
        self._y = np.array([])
        self._xlabel = ""
        self._ylabel = ""
        self._build()
        self._load_dataset()
        self._refit()

    # -- chrome -------------------------------------------------------------

    def _apply_palette(self) -> None:
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window, QColor(BG))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(FG))
        pal.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#1a1a1e"))
        pal.setColor(QPalette.ColorRole.Text, QColor(FG))
        pal.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(FG))
        pal.setColor(QPalette.ColorRole.Highlight, QColor(BAND))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(BG))
        self.setPalette(pal)
        self.setStyleSheet(
            f"""
            QWidget {{ background: {BG}; color: {FG}; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; font-size: 13px; }}
            QLabel#wordmark {{ font-family: 'Times New Roman', serif; font-size: 20px; }}
            QLabel#kicker {{ color: {MUTED}; font-size: 11px; letter-spacing: 1.4px; }}
            QLabel#hint {{ color: {MUTED}; font-size: 11px; }}
            QFrame#rail, QFrame#results, QFrame#overlay {{ background: {SURFACE}; }}
            QComboBox, QSlider, QTableWidget {{ background: {BG}; color: {FG}; }}
            QHeaderView::section {{ background: {SURFACE}; color: {MUTED}; border: none; padding: 6px; }}
            QTableWidget {{ gridline-color: {BORDER}; }}
            QCheckBox {{ spacing: 8px; }}
            QSplitter::handle {{ background: {BORDER}; }}
            QSplitter::handle:horizontal {{ width: 6px; }}
            QSplitter::handle:vertical {{ height: 6px; }}
            QPushButton, QToolButton {{
                background: {SURFACE}; color: {FG}; border: 1px solid {BORDER};
                border-radius: 6px; padding: 6px 10px;
            }}
            QPushButton:checked, QToolButton:checked {{ background: {BAND}; color: {BG}; }}
            """
        )

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header())

        self.h_split = QSplitter(Qt.Orientation.Horizontal)
        self.h_split.setChildrenCollapsible(True)
        self.v_split = QSplitter(Qt.Orientation.Vertical)
        self.v_split.setChildrenCollapsible(True)

        self.param_panel = self._params()
        self.chart_panel = self._chart()
        self.results_panel = self._results()

        self.v_split.addWidget(self.chart_panel)
        self.v_split.addWidget(self.results_panel)
        self.v_split.setStretchFactor(0, 3)
        self.v_split.setStretchFactor(1, 2)

        self.h_split.addWidget(self.param_panel)
        self.h_split.addWidget(self.v_split)
        self.h_split.setStretchFactor(0, 0)
        self.h_split.setStretchFactor(1, 1)
        self.h_split.setSizes([280, 1000])

        root.addWidget(self.h_split, 1)

        self.overlay = QFrame(self.chart_panel)
        self.overlay.setObjectName("overlay")
        self.overlay.setVisible(False)
        ov = QVBoxLayout(self.overlay)
        ov.setContentsMargins(0, 0, 0, 0)
        self.overlay_host = QWidget()
        ov.addWidget(self.overlay_host)

    def _header(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(52)
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 0, 16, 0)
        mark = QLabel("Fitline")
        mark.setObjectName("wordmark")
        self.subtitle = QLabel("OLS workbench")
        self.subtitle.setObjectName("hint")
        col = QVBoxLayout()
        col.setSpacing(0)
        col.addWidget(mark)
        col.addWidget(self.subtitle)
        row.addLayout(col)
        row.addStretch(1)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for key, label in (("split", "Split"), ("overlay", "Overlay"), ("chart", "Chart")):
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setChecked(key == "split")
            btn.clicked.connect(lambda _=False, k=key: self.set_layout_mode(k))
            self.mode_group.addButton(btn)
            row.addWidget(btn)
        return bar

    def _params(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("rail")
        rail.setMinimumWidth(220)
        wrap = QVBoxLayout(rail)
        wrap.setContentsMargins(16, 16, 16, 16)
        wrap.setSpacing(10)

        kicker = QLabel("SPECIFICATION")
        kicker.setObjectName("kicker")
        title = QLabel("Parameters")
        title.setObjectName("wordmark")
        wrap.addWidget(kicker)
        wrap.addWidget(title)

        wrap.addWidget(QLabel("Dataset"))
        self.dataset = QComboBox()
        self.dataset.addItems(list(DATASETS.keys()))
        self.dataset.currentIndexChanged.connect(self._on_dataset)
        wrap.addWidget(self.dataset)

        self.degree_label = QLabel("Polynomial degree  1")
        wrap.addWidget(self.degree_label)
        self.degree = QSlider(Qt.Orientation.Horizontal)
        self.degree.setRange(1, 5)
        self.degree.setValue(1)
        self.degree.valueChanged.connect(self._on_degree)
        wrap.addWidget(self.degree)

        self.intercept = QCheckBox("Intercept")
        self.intercept.setChecked(True)
        self.intercept.toggled.connect(self._refit)
        wrap.addWidget(self.intercept)

        self.robust = QCheckBox("Robust SEs (HC1)")
        self.robust.toggled.connect(self._refit)
        wrap.addWidget(self.robust)

        k2 = QLabel("CHART OVERLAYS")
        k2.setObjectName("kicker")
        wrap.addSpacing(8)
        wrap.addWidget(k2)
        self.show_fit = QCheckBox("Fitted curve")
        self.show_fit.setChecked(True)
        self.show_ci = QCheckBox("95% confidence band")
        self.show_ci.setChecked(True)
        self.show_pi = QCheckBox("95% prediction band")
        self.show_resid = QCheckBox("Residual stems")
        self.show_hist = QCheckBox("Residual histogram")
        self.show_hist.setChecked(True)
        for box in (self.show_fit, self.show_ci, self.show_pi, self.show_resid, self.show_hist):
            box.toggled.connect(self._draw)
            wrap.addWidget(box)

        hint = QLabel("Drag the splitters to resize panes. Overlay floats results on the chart.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        wrap.addSpacing(8)
        wrap.addWidget(hint)
        wrap.addStretch(1)
        return rail

    def _chart(self) -> QWidget:
        box = QFrame()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self.kicker_chart = QLabel("MAIN CHART")
        self.kicker_chart.setObjectName("kicker")
        self.chart_title = QLabel("y vs. x")
        self.chart_title.setObjectName("wordmark")
        layout.addWidget(self.kicker_chart)
        layout.addWidget(self.chart_title)
        self.figure = Figure(figsize=(6, 4), facecolor=BG)
        self.ax = self.figure.add_axes((0.10, 0.14, 0.86, 0.78))
        self.ax_hist = self.ax.inset_axes((0.72, 0.72, 0.26, 0.24))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, 1)
        return box

    def _results(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("results")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 12, 16, 12)
        kicker = QLabel("REGRESSION RESULTS")
        kicker.setObjectName("kicker")
        self.results_title = QLabel("OLS")
        self.results_title.setObjectName("wordmark")
        layout.addWidget(kicker)
        layout.addWidget(self.results_title)
        self.equation = QLabel("—")
        self.equation.setWordWrap(True)
        layout.addWidget(self.equation)
        self.stats = QLabel("")
        self.stats.setObjectName("hint")
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Term", "Est.", "SE", "t"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        return frame

    # -- behavior -----------------------------------------------------------

    def set_layout_mode(self, mode: str) -> None:
        self._layout_mode = mode
        if mode == "split":
            self.param_panel.setVisible(True)
            self.results_panel.setVisible(True)
            self.overlay.setVisible(False)
            self.h_split.setSizes([280, 1000])
            self.v_split.setSizes([400, 260])
        elif mode == "overlay":
            self.param_panel.setVisible(True)
            self.results_panel.setVisible(False)
            self.overlay.setVisible(True)
            self._place_overlay()
        else:
            self.param_panel.setVisible(False)
            self.results_panel.setVisible(False)
            self.overlay.setVisible(True)
            self._place_overlay()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.overlay.isVisible():
            self._place_overlay()

    def _place_overlay(self) -> None:
        host = self.chart_panel
        w, h = 380, int(host.height() * 0.46)
        self.overlay.setGeometry(host.width() - w - 16, host.height() - h - 16, w, h)
        if self.overlay_host.layout() is None:
            lay = QVBoxLayout(self.overlay_host)
            lay.setContentsMargins(0, 0, 0, 0)
        # Reparent a results clone visually by raising the existing panel contents:
        self.overlay.raise_()

    def _on_dataset(self) -> None:
        self._load_dataset()
        self._refit()

    def _on_degree(self, value: int) -> None:
        self.degree_label.setText(f"Polynomial degree  {value}")
        self._refit()

    def _load_dataset(self) -> None:
        name = self.dataset.currentText()
        x, y, xlabel, ylabel = DATASETS[name]()
        self._x, self._y = np.asarray(x, float), np.asarray(y, float)
        self._xlabel, self._ylabel = xlabel, ylabel
        self.chart_title.setText(f"{ylabel} vs. {xlabel}")
        self.subtitle.setText(f"{name} · {len(x)} observations")

    def _refit(self) -> None:
        fit = fit_ols(
            self._x,
            self._y,
            degree=int(self.degree.value()),
            intercept=self.intercept.isChecked(),
            robust=self.robust.isChecked(),
        )
        self._fit = fit
        self._fill_results(fit)
        self._draw()
        self.specChanged.emit()

    def _fill_results(self, fit: FitResult | None) -> None:
        if fit is None:
            self.equation.setText("Not enough observations to fit.")
            self.table.setRowCount(0)
            return
        deg = int(self.degree.value())
        extra = f" · degree {deg}" if deg > 1 else ""
        extra += " · HC1" if self.robust.isChecked() else ""
        self.results_title.setText(f"OLS{extra}")
        self.equation.setText(
            f"{fit.equation}\nz = (x − {fit.x_mean:.2f}) / {fit.x_sd:.2f}"
        )
        self.stats.setText(
            f"n {fit.n}    R² {fit.r2:.3f}    Adj. R² {fit.adj_r2:.3f}    "
            f"RMSE {fit.rmse:.3f}    F {fit.f_stat:.2f}    DW {fit.dw:.2f}"
        )
        self.table.setRowCount(len(fit.names))
        for i, name in enumerate(fit.names):
            vals = [name, f"{fit.beta[i]:.4g}", f"{fit.se[i]:.4g}", f"{fit.t[i]:.2f}"]
            for c, text in enumerate(vals):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignRight) | int(Qt.AlignmentFlag.AlignVCenter))
                self.table.setItem(i, c, item)

    def _draw(self) -> None:
        fit: FitResult | None = getattr(self, "_fit", None)
        ax, hist = self.ax, self.ax_hist
        ax.clear()
        hist = ax.inset_axes((0.72, 0.72, 0.26, 0.24))
        self.ax_hist = hist
        ax.set_facecolor(BG)
        hist.set_facecolor(SURFACE)
        self.figure.patch.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.set_xlabel(self._xlabel, color=MUTED, fontsize=9)
        ax.set_ylabel(self._ylabel, color=MUTED, fontsize=9)

        if fit is not None:
            if self.show_pi.isChecked():
                ax.fill_between(fit.x_grid, fit.pi_lo, fit.pi_hi, color=BAND, alpha=0.08, linewidth=0)
            if self.show_ci.isChecked():
                ax.fill_between(fit.x_grid, fit.ci_lo, fit.ci_hi, color=BAND, alpha=0.22, linewidth=0)
            if self.show_resid.isChecked():
                ax.vlines(self._x, fit.fitted, self._y, colors=RESIDUAL, alpha=0.45, linewidth=0.8)
            if self.show_fit.isChecked():
                ax.plot(fit.x_grid, fit.y_hat, color=FIT, lw=1.8, zorder=3)
            if self.show_hist.isChecked():
                hist.hist(fit.resid, bins=12, color=RESIDUAL, alpha=0.9)
                hist.tick_params(colors=MUTED, labelsize=6)
                hist.set_xticks([])
                hist.set_yticks([])
                for spine in hist.spines.values():
                    spine.set_color(BORDER)
            else:
                hist.set_visible(False)
        ax.scatter(self._x, self._y, s=18, c=POINT, alpha=0.82, zorder=4, linewidths=0)
        self.canvas.draw_idle()


def main() -> int:
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Fitline")
    window.setCentralWidget(RegressionWorkbench())
    window.resize(1280, 820)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
