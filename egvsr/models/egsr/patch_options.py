import torch


def same_padding(images, kernel_size, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (kernel_size[0] - 1) * rates[0] + 1
    effective_k_col = (kernel_size[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_patches(images, kernel_sizes, strides, rates):
    """
    Extract patches from images and put them in the C output dimension.
    e.g. [b, c, h, w] -> [b, c*k*k, h*w]
    """
    assert len(images.size()) == 4
    images = same_padding(images, kernel_sizes, strides, rates)
    unfold = torch.nn.Unfold(kernel_size=kernel_sizes, dilation=rates, padding=0, stride=strides)
    patches = unfold(images)
    return patches


def restore_patches(images, out_size, ksizes, strides, padding):
    """
    Extract patches from images and put them in the C output dimension.
    e.g. [b, c*k*k, h*w] -> [b, c, h, w]
    """
    unfold = torch.nn.Fold(
        output_size=out_size,
        kernel_size=ksizes,
        dilation=1,
        padding=padding,
        stride=strides,
    )
    patches = unfold(images)
    return patches
