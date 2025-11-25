import numpy as np
from PIL import Image

def histogram_matching(source_image, reference_image):
    """
    Perform histogram matching on the source image using the reference image.

    Parameters:
        source_image (str or numpy.ndarray): Path to the source image or source image as numpy array.
        reference_image (str or numpy.ndarray): Path to the reference image or reference image as numpy array.

    Returns:
        numpy.ndarray: The matched image as numpy array.
    """
    # Load the images
    if isinstance(source_image, str):
        source_image = np.array(Image.open(source_image))
    if isinstance(reference_image, str):
        reference_image = np.array(Image.open(reference_image))

    # Get the histogram of the source and reference images
    source_hist, _ = np.histogram(source_image.flatten(), bins=256, range=(0, 255))
    reference_hist, _ = np.histogram(reference_image.flatten(), bins=256, range=(0, 255))

    # Compute the cumulative distribution functions (CDFs) of the histograms
    source_cdf = np.cumsum(source_hist) / float(source_image.size)
    reference_cdf = np.cumsum(reference_hist) / float(reference_image.size)

    # Compute the mapping from the source histogram to the reference histogram
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 255
        while j >= 0 and source_cdf[i] <= reference_cdf[j]:
            j -= 1
        mapping[i] = j

    # Apply the mapping to the source image
    matched_image = mapping[source_image]
    matched_image = np.clip(matched_image, 0, 255)

    return matched_image

if __name__ == '__main__':
    # Perform histogram matching on the source image using the reference image
    matched_image = histogram_matching('test_data/Norm/lowlight/a2221-dvf_098.jpg', 'test_data/Norm/high/a2221-dvf_098.jpg')

    # Save the matched image
    Image.fromarray(matched_image).save('test_data/matched.jpg')