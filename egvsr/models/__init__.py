from egvsr.models.rssrt.random_scale_super_resolution_with_event import (
    RandomScaleSuperResolutionWithEvent,
)


def get_model(config):
    if config.NAME == "RSSR":
        image_size = (config.image_size[0], config.image_size[1])
        return RandomScaleSuperResolutionWithEvent(
            in_frames=config.in_frames,
            out_frames=config.out_frames,
            is_include_bound=config.is_include_bound,
            moments=config.moments,
            event_channels=config.event_channels,
            image_size=image_size,
            channels=config.channels,
            n_feats=config.n_feats,
            patch_size=config.patch_size,
            is_shallow_fusion=config.is_shallow_fusion,
            sr_low_scale=config.sr_low_scale,
            sr_up_scale=config.sr_up_scale,
            interp_mode=config.interp_mode,
            random_up_sampler=config.random_up_sampler,
            time_bins=config.time_bins,
            inr_channel=config.inr_channel,
            shallow_cnn_depth=config.shallow_cnn_depth,
            shallow_transformer_layer=config.shallow_transformer_layer,
            deep_cnn_depth=config.deep_cnn_depth,
            deep_transformer_layer=config.deep_transformer_layer,
            event_residual_connection=config.event_residual_connection,
            event_residual_sample_number=config.event_residual_sample_number,
            event_residual_offset=config.event_residual_offset,
            event_residual_layers=config.event_residual_layers,
            has_event_reconstruction=config.has_event_reconstruction,
        )
    else:
        raise ValueError(f"Unknown model: {config.NAME}")
