import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from matplotlib import cm
from scipy import ndimage
from tqdm import tqdm
import string



def is_empty_section(image, brightness_threshold=100, cnt_threshold=0.05, plot=False):
    """
    Checks if a split image is mostly dark and free of bright objects.

    Parameters:
    - image: Input frame (numpy array).
    - brightness_threshold: Max brightness to be considered "dark."
    - max_bright_ratio: Max percentage of bright pixels allowed.

    Returns:
    - True if the frame is good, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histo_dict = histogram(
        gray,
        #cv2.GaussianBlur(gray, (5, 5), 0),
        plot=plot)
    score = sum([v for k, v in histo_dict.items() if k>brightness_threshold])

    return score # Return True if mostly dark

def histogram(gray, plot=False):
    """
    Compute the histogram of a grayscale image.
    :param gray: grayscale image
    :param plot: (bool)
    :return: (dict) histogram
    """
    hist, bins = np.histogram(gray, bins=26, range=(0, 255))

    if plot:
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot grayscale image
        axes[0].imshow(gray, cmap='gray')
        axes[0].set_title("Grayscale Image")
        axes[0].axis("off")  # Hide axes

        # Plot histogram
        axes[1].plot(bins[:-1], hist, color='black')
        axes[1].set_title("Brightness Histogram")
        axes[1].set_xlabel("Normalized Brightness (0=Dark, 1=Bright)")
        axes[1].set_ylabel("Pixel Count")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    tot = sum(hist)
    histo_dict = {k: n / tot for n, k in zip(hist, bins[:-1])}
    return histo_dict

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


def sobel_filter(image):
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient
    magnitude = np.sqrt(sobel_h ** 2 + sobel_v ** 2)
    magnitude *= 255.0 / np.max(magnitude)  # normalization
    return magnitude

def extract_frames(video_path, output_folder, frame_interval=5, resize=False):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save extracted frames.
    - frame_interval: Save one frame every 'frame_interval' frames.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    os.makedirs(output_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.png"
            if resize:
                frame = cv2.resize(frame, (256, 256))
            cv2.imwrite(os.path.join(output_folder, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_folder}")
    pass


def extract_semiframes(video_path, output_folder, frame_interval=5):
    """
    Extracts semi-frames from a video, filer the empty ones and saves them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save extracted frames.
    - frame_interval: Save one frame every 'frame_interval' frames.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    for i in range(4):
        os.makedirs(os.path.join(output_folder, f"sf_{i}"), exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            filename = f"semiframe_{saved_count:04d}.png"
            splitted_frames = split_frames(frame)
            for i, f in enumerate(splitted_frames):
                score = is_empty_section(f, cnt_threshold=0.06, plot=False)
                if score > 0.02 and score < 0.063:
                    cv2.imwrite(os.path.join(output_folder, f"sf_{i}", filename), f)
                    saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_folder}")


def assemble_frames(n, semiframes_path, output_folder, overlap=3, resize=False):
    folders = {
        "top_left": "sf_0",
        "top_right": "sf_1",
        "bottom_left": "sf_2",
        "bottom_right": "sf_3"
    }
    os.makedirs(output_folder, exist_ok=True)
    semiframes = {
        k: os.listdir(semiframes_path + f'/{v}')
        for k, v in folders.items()
    }

    items = range(n)

    for i in tqdm(items, desc="Processing items", unit="item"):
        # Randomly select one image from each folder
        selected_images = {
            key: random.choice(files)
            for key, files in semiframes.items()
        }

        # Open selected images
        top_left = Image.open(os.path.join(semiframes_path, folders["top_left"], selected_images["top_left"]))
        top_right = Image.open(os.path.join(semiframes_path, folders["top_right"], selected_images["top_right"]))
        bottom_left = Image.open(os.path.join(semiframes_path, folders["bottom_left"], selected_images["bottom_left"]))
        bottom_right = Image.open(
            os.path.join(semiframes_path, folders["bottom_right"], selected_images["bottom_right"]))

        # Blend top-left and top-right horizontally
        top_half = blend_images(top_left, top_right, overlap, "horizontal")

        # Blend bottom-left and bottom-right horizontally
        bottom_half = blend_images(bottom_left, bottom_right, overlap, "horizontal")

        # Blend top and bottom halves vertically
        final_image = blend_images(top_half, bottom_half, overlap, "vertical")
        if resize:
            fi_ = cv2.resize(np.asarray(final_image), (256, 256))
            final_image = Image.fromarray(fi_)

        #img_name = f"image_{i + 1}"
        img_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        final_image.save(os.path.join(output_folder, img_name + ".png"))


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

