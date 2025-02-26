import matplotlib

matplotlib.use('TkAgg')  # This will force matplotlib to use an interactive backend
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore
from torchvision import transforms
import os
import torch
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
from anomalib.utils.post_processing import superimpose_anomaly_map
from tqdm import tqdm
from pathlib import Path

size = 512 #(256, 256)

def preprocess_data(dataset_path, normal_folder, abnormal_dir, batch_size):

    print('Process data')

    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])

    # load the dataset in a datamodule
    datamodule = Folder(
        name='profile',
        root=dataset_path,
        normal_dir=normal_folder,
        abnormal_dir=abnormal_dir,
        task="classification",
        image_size=size,
        num_workers=0,
        train_batch_size=batch_size,
        eval_batch_size=4,
        transform=transform,
    )
    datamodule.setup()

    return datamodule


def test(task_name, weights_path, test_folder):
    print('Begin testing')
    model = Patchcore()
    model.load_state_dict(torch.load(os.path.join(weights_path, f'patchcore_checkpoint_{task_name}.pt')))

    # load the engine
    engine = Engine(task="classification")

    # predict make predictions
    with torch.no_grad():
        predictions = engine.predict(
            model=model,
            return_predictions=True,
            data_path=test_folder)
    return predictions


def train(
        task_name,
        dataset_path,
        weights_path,
        save_weights=True,
        train_folder='train/OK',
        eval_folder='val',
        batch_size=32
    ):

    training_datamodule = preprocess_data(
        dataset_path,
        normal_folder=train_folder,
        abnormal_dir=eval_folder,
        batch_size=batch_size
    )

    print('Begin training')
    model = Patchcore()
    engine = Engine(task="classification")
    engine.fit(model=model, datamodule=training_datamodule)

    if save_weights:
        os.makedirs(weights_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(weights_path, f'patchcore_checkpoint_{task_name}_{size}.pt'))

    engine.test(datamodule=training_datamodule, model=model)


def get_segment(original_image:np.ndarray, heatmap_image:np.ndarray, threshold:float):
    mask = (heatmap_image ** 2 + 0.2 > threshold)
    return Image.fromarray((original_image * mask).astype(np.uint8))


def get_heatmap(original_image:np.ndarray, heatmap_image:np.ndarray):
    anomaly_map = np.array((1. - heatmap_image) * 255).astype(np.uint8)
    heat_img = superimpose_anomaly_map(anomaly_map=anomaly_map, image=img, normalize=False)
    return Image.fromarray(heat_img.astype(np.uint8))


def get_detect(original_image:np.ndarray, boxes:list, score:float):
    # superimpose bounding boxes
    for (x1, y1, x2, y2) in boxes:
        detect_img = cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    detect_img = Image.fromarray(detect_img.astype(np.uint8))
    # add score
    draw = ImageDraw.Draw(detect_img)
    font = ImageFont.load_default()
    text = f"anomaly score {round(score * 100, 1)}%"
    position = (10, 5)  # X, Y coordinates
    draw.text(position, text, font=font, fill=(255, 0, 0))
    return detect_img


def process_frames(
        task_name: str,
        weights_path: Path,
        dataset_path: Path,
        segment_path: Path,
        heatmap_path: Path,
        detect_path: Path,
        segmentation: bool=True,
        detection: bool=True,
        heatmapping: bool=True,
        threshold=0.55,
    ) -> None:

    model = Patchcore()
    model.load_state_dict(torch.load(os.path.join(weights_path, f'patchcore_checkpoint_{task_name}.pt')))

    engine = Engine(task="classification")
    predictions = engine.predict(
        model=model,
        return_predictions=True,
        data_path=dataset_path)

    items = range(len(predictions))

    for i in tqdm(items, desc="Processing items", unit="item"):
        pred = predictions[i]
        heatmap = np.transpose(pred['anomaly_maps'].squeeze(0).numpy(), (1, 2, 0))
        img = cv2.imread(pred['image_path'][0])
        img = cv2.resize(img, heatmap.shape[:2])
        frame_name = pred['image_path'][0].split('/')[-1]

        if segmentation:
            os.makedirs(segment_path, exist_ok=True)
            segment_img = get_segment(img, heatmap, threshold)
            segment_img.save(os.path.join(segment_path, frame_name))

        if heatmapping:
            os.makedirs(heatmap_path, exist_ok=True)
            heat_img = get_heatmap(img, heatmap)
            heat_img.save(os.path.join(heatmap_path, frame_name))

        if detection:
            os.makedirs(detect_path, exist_ok=True)
            detect_img = get_detect(
                original_image=img,
                boxes=pred['pred_boxes'][0].int().tolist(),
                score=pred['pred_scores'][0].item()
            )
            detect_img.save(os.path.join(detect_path, frame_name))

