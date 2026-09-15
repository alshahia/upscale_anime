"""Extracted UI sections. Each widget is a self-contained Frame/LabelFrame that owns
its variables and child widgets. The orchestrator (UpscaleGUI) composes them.

Design notes:
  - Widgets are intentionally _underscore-prefixed to mark them as internal to the
    GUI subpackage. They are not part of any public API.
  - Each widget takes (parent, app) and stores `self.app` for callbacks into the
    orchestrator. Avoid passing type hints on `app` to prevent import cycles.
  - Widgets never read each other; all cross-widget coordination happens in app.py.
"""
from .empty_state import _EmptyState
from .gpu_monitor import _GPUMonitor
from .input_panel import _InputPanel
from .log_panel import _LogPanel, _LogPanelHandler
from .menu_bar import _MenuBar
from .model_panel import _ModelPanel
from .settings_panel import _SettingsPanel
from .output_panel import _OutputPanel
from .models_panel import _ModelsPanel
from .status_bar import _StatusBar
from .splash import Splash
from .toast import Toast
