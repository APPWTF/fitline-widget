# Fitline widget

Drop-in Qt OLS workbench (`QWidget`). Not TypeScript.

```bash
pip install -r requirements.txt
python fitline_widget.py
```

Embed in another window:

```python
from PySide6.QtWidgets import QMainWindow
from fitline_widget import RegressionWorkbench

window = QMainWindow()
window.setCentralWidget(RegressionWorkbench())
window.resize(1280, 820)
window.show()
```
