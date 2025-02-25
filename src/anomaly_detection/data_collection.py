import cv2
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import string
from anomaly_detection.image_process import blend_images, split_frames
from anomaly_detection.utils import histogram


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
        plot=plot)
    score = sum([v for k, v in histo_dict.items() if k>brightness_threshold])
    return score # Return True if mostly dark


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
    '''
    Assemble image-frames with the combination of the semiframes in the folders.
    :param n:
    :param semiframes_path:
    :param output_folder:
    :param overlap:
    :param resize:
    :return:
    '''
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
        # randomly select one image from each folder
        selected_images = {
            key: random.choice(files)
            for key, files in semiframes.items()
        }

        # open selected images
        top_left = Image.open(os.path.join(semiframes_path, folders["top_left"], selected_images["top_left"]))
        top_right = Image.open(os.path.join(semiframes_path, folders["top_right"], selected_images["top_right"]))
        bottom_left = Image.open(os.path.join(semiframes_path, folders["bottom_left"], selected_images["bottom_left"]))
        bottom_right = Image.open(
            os.path.join(semiframes_path, folders["bottom_right"], selected_images["bottom_right"]))

        # blend top-left and top-right horizontally
        top_half = blend_images(top_left, top_right, overlap, "horizontal")

        # blend bottom-left and bottom-right horizontally
        bottom_half = blend_images(bottom_left, bottom_right, overlap, "horizontal")

        # blend top and bottom halves vertically
        final_image = blend_images(top_half, bottom_half, overlap, "vertical")
        if resize:
            fi_ = cv2.resize(np.asarray(final_image), (256, 256))
            final_image = Image.fromarray(fi_)

        #img_name = f"image_{i + 1}"
        img_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        final_image.save(os.path.join(output_folder, img_name + ".png"))




