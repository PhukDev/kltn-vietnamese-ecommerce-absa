from __future__ import annotations

import importlib.abc
import sys


class _OptionalPyArrowBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError("pyarrow is disabled for this baseline run")
        return None


def disable_optional_pyarrow() -> None:
    """Keep scikit-learn from loading pyarrow when Windows blocks its DLL.

    scikit-learn only uses pyarrow for optional compatibility checks in this
    pipeline. Hiding it avoids a Windows Application Control DLL failure without
    affecting CSV training.
    """

    if any(isinstance(finder, _OptionalPyArrowBlocker) for finder in sys.meta_path):
        return
    sys.modules.pop("pyarrow", None)
    sys.meta_path.insert(0, _OptionalPyArrowBlocker())
