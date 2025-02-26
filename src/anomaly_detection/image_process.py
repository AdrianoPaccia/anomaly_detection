import numpy as np
from PIL import Image
import cv2

def blend_images(
        image1: Image,
        image2: Image,
        overlap: float,
        direction="horizontal",
        blur=True
    ) -> Image:
    """
    Blend two images with a smooth transition along the given direction.

    :param image1: first image
    :param image2: second image
    :param overlap: (float) dimention of the overlapping section
    :param direction: (horizontal, vertical) direction of the image stitching
    :param blur: (bool) to use a Gaussian Blur filter on the overlapping section
    :return: final blended image
    """
    width, height = image1.size
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


def get_gradient_mask(size, direction="horizontal", blending_mode="exponential") -> np.ndarray:
    """
    Compute the gradient mask for a given size and direction.

    :param size: (int) size of the section to compute the gradient mask on
    :param direction: (horizontal, vertical) direction of the gradient
    :param blending_mode:  (linear, quadratic, cubic, exponential) mode of blending
    :return:
    """
    width, height = size
    gradient = np.expand_dims(
        np.array(
            [blend_law(i, width, mode=blending_mode) for i in range(width)]
        ), -1
    )
    mask = np.ones((height, width, 1)) * gradient
    if direction == "horizontal":
        return mask
    else:
        return np.transpose(mask, (1, 0, 2))


def blend_law(idx, tot, mode="linear"):
    if mode == "linear":
        return 1. - idx / tot
    elif mode == "quadratic":
        return 1. - (idx / tot) ** 2
    elif mode == "cubic":
        return 1. - (idx / tot) ** 3
    elif mode == "exponential":
        return np.exp(-idx / tot)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def split_frames(image):
    h, w = image.shape[:2]
    mid_w = w // 2  # Midpoint for splitting
    mid_h = h // 2
    f1, f2, f3, f4 = image[:mid_h, :mid_w], image[:mid_h, mid_w:], image[mid_h:, :mid_w], image[mid_h:, mid_w:]
    return f1, f2, f3, f4