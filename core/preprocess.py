import numpy as np
from scipy import ndimage


def mask_padding(
    array: np.ndarray, threshold: int = 10, dark: int = 128, min_frac: float = 0.005
) -> np.ndarray:
    """Mark warp padding (near-black, corner-connected) as invalid.

    Only floods from *dark* corners, and ignores small floods, so this is correct
    whether called on the full TIFF (black corner triangles -> padding) or on an
    interior ROI crop (bright corners -> nothing flooded -> all valid). Calling it
    on the ROI instead of the full image avoids flood-filling tens of Mpx of
    padding that the ROI never uses.

    Args:
        array: Grayscale image (full TIFF or an ROI crop).
        threshold: Max intensity difference from the seed for a pixel to count as
            padding during the flood.
        dark: Corners brighter than this are treated as paper, not padding, and
            are not used as flood seeds.
        min_frac: Floods smaller than this fraction of the image are ignored
            (e.g. a parcel line that happens to touch a corner).

    Returns:
        Boolean mask, same shape as ``array``; True for valid (non-padding) pixels.
    """
    h, w = array.shape
    seeds = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    padding = np.zeros(array.shape, dtype=bool)
    arr_f = array.astype(float)
    for seed in seeds:
        seed_value = float(array[seed])
        if seed_value >= dark:
            continue  # bright corner: paper, not padding
        candidate = np.abs(arr_f - seed_value) <= threshold
        marker = np.zeros(array.shape, dtype=bool)
        marker[seed] = True
        flood = ndimage.binary_propagation(
            marker, structure=np.ones((3, 3), dtype=bool), mask=candidate
        )
        if flood.sum() >= min_frac * array.size:  # ignore a line through a corner
            padding |= flood
    return ~padding
