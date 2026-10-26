"""实验进程的资源约束：单线程数学库 + 低优先级（多进程 × 多线程会过度订阅 CPU，且机器上有其他任务）。

**必须在 numpy / cv2 之前 import**（环境变量在库加载时读取）；子进程（spawn）继承这里设的环境变量，
另在 `init_worker` 里对 OpenCV 设单线程并降低优先级。线程数不应改变数值结果（逐项核对见
docs/EXPERIMENTS.md）。
"""

from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

BELOW_NORMAL = 0x4000


def lower_priority():
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415

        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL)
    else:  # pragma: no cover
        os.nice(10)


def init_worker():
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
    lower_priority()


init_worker()  # 主进程也一样
