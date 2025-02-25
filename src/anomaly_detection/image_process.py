import numpy as np
from PIL import Image
import cv2
from scipy import ndimage

def blend_images(image1, image2, overlap, direction="horizontal", blur=True):
    """ Blend two images with a smooth transition along the given direction. """
    width, height = image1.size
    new_size = (width * 2, height) if direction == "horizontal" else (width, height * 2)
    img1, img2 = np.array(image1), np.array(image2)

    if direction == "horizontal":
        padd1, padd2 = img1[:, -overlap * 2:], img2[:, :overlap * 2]
        img1_, img2_ = img1[:, :-overlap], img2[:, overlap:]
        mask = get_gradient_mask((overlap * 2, height), "horizontal")
        over_part = padd1 * mask + padd2 * (1 - mask)
        if blur:
            over_part = cv2.GaussianBlur(over_part.astype(np.uint8), (5, 5), 0)
        blended = np.hstack((img1_, over_part, img2_))

    else:
        padd1, padd2 = np.flip(img1[-overlap * 2:, :], axis=0), np.flip(img2[:overlap * 2, :], axis=0)
        img1_, img2_ = img1[:-overlap, :], img2[overlap:, :]
        mask = get_gradient_mask((overlap * 2, width), "vertical")
        over_part = padd1 * mask + padd2 * (1 - mask)
        if blur:
            over_part = cv2.GaussianBlur(over_part.astype(np.uint8), (5, 5), 0)
        blended = np.vstack((img1_, over_part, img2_))

    final_img = blended.astype(np.uint8)
    return Image.fromarray(final_img)


def blend_law(i, n, mode="linear"):
    if mode == "linear":
        return 1. - i / n
    elif mode == "quadratic":
        return 1. - (i / n) ** 2
    elif mode == "cubic":
        return 1. - (i / n) ** 3
    elif mode == "exponential":
        return np.exp(-i / n)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_gradient_mask(size, direction="horizontal"):
    width, height = size
    gradient = np.expand_dims(
        np.array(
            [blend_law(i, width, "exponential") for i in range(width)]
        ), -1
    )
    mask = np.ones((height, width, 1)) * gradient
    if direction == "horizontal":
        return mask
    else:
        return np.transpose(mask, (1, 0, 2))


def sobel_filter(image):
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    magnitude *= 255.0 / np.max(magnitude)  # normalization
    return magnitude


def split_frames(image):
    """
    Splits an image into left and right halves and saves them separately.

    Parameters:
    - image: Input frame (numpy array).
    """
    h, w = image.shape[:2]
    mid_w = w // 2  # Midpoint for splitting
    mid_h = h // 2
    f1, f2, f3, f4 = image[:mid_h, :mid_w], image[:mid_h, mid_w:], image[mid_h:, :mid_w], image[mid_h:, mid_w:]
    return f1, f2, f3, f4