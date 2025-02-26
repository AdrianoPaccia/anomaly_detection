import cv2
import os
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import string
from anomaly_detection.image_process import blend_images, split_frames
from anomaly_detection.utils import histogram, get_histo_distance
from pathlib import Path

def extract_frames(video_path, output_folder, frame_interval=5, save=False):
    """
    Extracts frames from a video and saves them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save extracted frames.
    - frame_interval: Save one frame every 'frame_interval' frames.
    """
    print(f'Extracting frames from {video_path}...')
    cap = cv2.VideoCapture(video_path)
    frame_count, saved_count = 0, 0
    saved_images = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.png"

            if save:
                os.makedirs(output_folder, exist_ok=True)
                cv2.imwrite(os.path.join(output_folder, filename), frame)
            else:
                saved_images[filename] = frame
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames")
    return saved_images


def extract_empty_semiframes(
        video_path: Path,
        output_path: Path,
        empty_sample: Path,
        frame_interval=5
    ) -> None:
    """
    Extracts semi-frames from a video, filer the empty ones and saves them as images.

    Parameters:
    - video_path: Path to the input video file.
    - output_folder: Folder to save extracted frames.
    - frame_interval: Save one frame every 'frame_interval' frames.
    """
    os.makedirs(output_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    for i in range(4):
        os.makedirs(os.path.join(output_path, f"sf_{i}"), exist_ok=True)

    #analize empty sample
    empty_frame = Image.fromarray(cv2.imread(empty_sample))
    empty_histo = histogram(empty_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0:
            filename = f"semiframe_{saved_count:04d}.png"
            splitted_frames = split_frames(frame)
            for i, f in enumerate(splitted_frames):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                img_histo = histogram(gray, plot=False)
                if get_histo_distance(empty_histo, img_histo) < 0.18:
                    cv2.imwrite(os.path.join(output_path, f"sf_{i}", filename), f)
                    saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames and saved to {output_path}")


def assemble_semiframes(n, semiframes_path, output_folder, overlap=3):
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

    for _ in tqdm(range(n), desc="Processing items", unit="item"):
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

        img_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        final_image.save(os.path.join(output_folder, img_name + ".png"))




