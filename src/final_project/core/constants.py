"""Common constants used throughout the unified cutter.

These values control the minimum and maximum durations of clips,
the target sizes for vertical output videos and the expected
source resolution.  They can be adjusted centrally here.
"""

from typing import Tuple

# Minimum duration (in seconds) for a candidate clip.  This value
# determines the lower bound for both speech and beat based clips.
MIN_CLIP_DURATION: int = 15

# Maximum duration (in seconds) for a candidate clip.  If a segment
# exceeds this duration it will be truncated or divided.
MAX_CLIP_DURATION: int = 20

# Minimum number of clips to produce when iteratively selecting from
# speech segments.  The selection algorithm will try to reach this
# number if enough candidates are available.
MIN_NUM_CLIPS: int = 50

# Maximum number of clips to produce.  This protects against runaway
# generation on long videos.
MAX_NUM_CLIPS: int = 60

# Expected horizontal source resolution (width, height).  If an
# incoming clip does not match this resolution it will be resized
# before being further processed.  CutterPy expects 1920×1080 by
# default.
EXPECTED_SHORT_SOURCE_SIZE: Tuple[int, int] = (1920, 1080)

# Target vertical output resolution (width, height) for TikTok/Shorts
# style videos.  Clips will be scaled to this size and padded
# appropriately.
TARGET_SHORT_SIZE: Tuple[int, int] = (1080, 1920)
STANDARD_FOREGROUND_SCALE: float = 1.2
STANDARD_BACKGROUND_BLUR_KERNEL: int = 81

# Alternative target used by the dual frame editor.  When two
# horizontal crops are stacked on top of one another the height of
# each part is TARGET_SHORT_SIZE_NEW.  See EditorPoint for details.
TARGET_SHORT_SIZE_NEW: Tuple[int, int] = (1080, 960)
