"""
src/__init__.py
Exposes library version info for reproducibility logging.
"""
import torch, torchvision, sklearn, pandas, numpy

VERSIONS = {
    "torch":       torch.__version__,
    "torchvision": torchvision.__version__,
    "sklearn":     sklearn.__version__,
    "pandas":      pandas.__version__,
    "numpy":       numpy.__version__,
}
