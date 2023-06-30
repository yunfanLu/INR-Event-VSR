from egvsr.datasets.alpx_vsr_dataset import get_alpx_vsr_dataset
from egvsr.datasets.color_event_dataset import (
    get_ced_dataset,
)


def get_dataset(config):
    if config.NAME == "CED":
        return get_ced_dataset(
            config.CED_ROOT,
            config.IN_FRAME,
            config.FUTURE_FRAME,
            config.PAST_FRAME,
            config.SCALE,
            config.MOMENTS,
            config.is_mini,
        )
    elif config.NAME == "ALPX":
        return get_alpx_vsr_dataset(
            alpx_vsr_root=config.ALPX_VSR_ROOT,
            moments=config.MOMENTS,
            in_frame=config.IN_FRAME,
            future_frame=config.FUTURE_FRAME,
            past_frame=config.PAST_FRAME,
            scale=config.SCALE,
            random_crop_resolution=config.RANDOM_CROP_RESOLUTION,
            high_resolution=config.HIGH_RESOLUTION,
            low_resolution=config.LOW_RESOLUTION,
            evaluation_visualization=config.EVALUATION_VISUALIZATION,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")
