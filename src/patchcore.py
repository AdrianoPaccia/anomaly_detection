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

size = 512 #(256, 256)

def save_model_weights(model, path, checkpoint_name='patchcore_checkpoint.pt'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), str(Path(path) / checkpoint_name))

def load_model_weights(model, path, checkpoint_name='patchcore_checkpoint.pt'):
    model.load_state_dict(torch.load(str(Path(path) / checkpoint_name)))
def preprocess_data(dataset_path, normal_folder, abnormal_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    # Load our dataset in a datamodule for the model to train on
    datamodule = Folder(
        name='profile',
        root=dataset_path,
        normal_dir=normal_folder,  # Path to the anomaly free images (for training)
        abnormal_dir=abnormal_dir,  # Path to the anomalous images (for validation)
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
    load_model_weights(model, weights_path, f'patchcore_checkpoint_{task_name}.pt')

    engine = Engine(task="classification")

    # Predict
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
        eval_folder='val/NG',
        batch_size=32
    ):
    print('Process data')
    training_datamodule = preprocess_data(
        dataset_path,
        normal_folder=train_folder,
        abnormal_dir=eval_folder,
        batch_size=batch_size
    )

    print('Begin training')
    # Define the model and train with our dataset
    model = Patchcore()
    # Initialize the engine with classification as task
    engine = Engine(task="classification")
    # engine = Engine()
    engine.fit(model=model, datamodule=training_datamodule)

    if save_weights:
        save_model_weights(model, weights_path, f'patchcore_checkpoint_{task_name}_{size}.pt')

    engine.test(datamodule=training_datamodule, model=model)


def process_frames(
        task_name,
        dataset_path,
        weights_path,
        segment_path,
        heatmap_path,
        detect_path,
        segmentation=True,
        detection=False,
        heatmapping=False,
        threshold=0.55
    ):
    model = Patchcore()
    load_model_weights(model, weights_path, f'patchcore_checkpoint_{task_name}.pt')
    engine = Engine(task="classification")
    predictions = engine.predict(
        model=model,
        return_predictions=True,
        data_path=dataset_path)

    items = range(len(predictions))

    for i in tqdm(items, desc="Processing items", unit="item"):
        pred = predictions[i]
        img = cv2.imread(pred['image_path'][0])
        heatmap = np.transpose(pred['anomaly_maps'].squeeze(0).numpy(), (1, 2, 0))
        img = cv2.resize(img, heatmap.shape[:2])
        frame_name = pred['image_path'][0].split('/')[-1]

        if segmentation:
            os.makedirs(segment_path, exist_ok=True)
            mask = (heatmap**2+0.2 > threshold).astype(np.uint8)
            segment_img = Image.fromarray((img*mask).astype(np.uint8))
            segment_img.save(os.path.join(segment_path, frame_name))

        if heatmapping:
            os.makedirs(heatmap_path, exist_ok=True)
            anomaly_map = np.array((1.-heatmap)*255).astype(np.uint8)

            # Superimpose the anomaly map onto the original image
            heat_img = superimpose_anomaly_map(anomaly_map=anomaly_map, image=img, normalize=False)
            heat_img = Image.fromarray(heat_img.astype(np.uint8))
            heat_img.save(os.path.join(heatmap_path, frame_name))

        if detection:
            os.makedirs(detect_path, exist_ok=True)
            boxes = pred['pred_boxes'][0].int().tolist()
            detect_img = img
            #superimpose bounding boxes
            for (x1, y1, x2, y2) in boxes:
                detect_img = cv2.rectangle(detect_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            detect_img = Image.fromarray(detect_img.astype(np.uint8))
            #add score
            draw = ImageDraw.Draw(detect_img)
            #font = ImageFont.truetype("arial.ttf", 12)  # You can use a custom font file
            font = ImageFont.load_default()
            text = f"anomaly score {round(pred['pred_scores'][0].item() * 100, 1)}%"
            position = (10, 5)  # X, Y coordinates
            draw.text(position, text, font=font, fill=(255, 0, 0))  # Yellow text
            #save
            detect_img.save(os.path.join(detect_path, frame_name))

    return predictions

