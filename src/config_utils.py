"""Configuration normalization without importing GPU frameworks."""
import math


def image_score_thresholds(config):
    """Thresholds in the unified class order, with nested-over-flat precedence."""
    count = config.get('num_classes', len(config.get('classes', [])) or 10)
    value = config.get('detector2d', {}).get('score_thr', config.get('score_thr_2d', 0.5))
    if isinstance(value, (int, float)):
        value = [float(value)] * count
    if len(value) != count:
        raise ValueError('2D score thresholds must match the unified class count')
    values = [float(v) for v in value]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('2D score thresholds must be finite values in [0,1]')
    return values
