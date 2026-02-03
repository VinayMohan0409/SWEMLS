from __future__ import annotations

# Ensure the repository root is importable when running `pytest`.
# Some environments/run configurations do not automatically add the project root
# to sys.path, which can cause `import aki_service` to fail.

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
