"""pytest 公共设置：把 src/ 加入 import 路径。

这样测试里可以直接 ``from data import ...``、``from models import ...``，
和 train.py 的 import 方式一致。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
