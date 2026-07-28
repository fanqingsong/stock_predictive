"""Stock prediction ML package (scale-native day/week/month classifiers)."""

from .timeframes import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    FEATURE_VERSION,
    TASK_TYPE,
    TIMEFRAME_KEYS,
    TIMEFRAMES,
    TRAIN_RATIO,
    max_history_bars,
    timeframe_config,
)

# Backward-compatible aliases used by routes/worker.
LOOKBACK = TIMEFRAMES["day"]["lookback"]
