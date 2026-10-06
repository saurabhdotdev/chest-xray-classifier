"""
src.ui.pages
============
Workstation page modules for ChestAI.
"""

from src.ui.pages.home import page_home
from src.ui.pages.analyze import page_analyze
from src.ui.pages.compare import page_compare
from src.ui.pages.triage import page_triage
from src.ui.pages.performance import page_performance
from src.ui.pages.calibration import page_calibration
from src.ui.pages.shift import page_shift
from src.ui.pages.about import page_about

__all__ = [
    "page_home",
    "page_analyze",
    "page_compare",
    "page_triage",
    "page_performance",
    "page_calibration",
    "page_shift",
    "page_about",
]
