import argparse
from anomaly_detection.patchcore import test, train, process_frames
from anomaly_detection.data_collection import extract_frames
from anomaly_detection.utils import merge
import matplotlib
matplotlib.use('TkAgg')
import os
from pathlib import Path
import yaml

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", required=True, type=str, help="Name of task (dataset).")
    args.add_argument('--train', action='store_true', help="Training required.")
    args.add_argument('--test', action='store_true', help="Testing required.")
    args.add_argument('--segment', action='store_true', help="Segmentation required.")
    args.add_argument('--detect', action='store_true', help="Classification required.")
    args.add_argument('--heatmap', action='store_true', help="Heatmapping required.")

    #HYPERPARAMS
    args.add_argument('--threshold', type=float, help="Plotting required.")
    args.add_argument("--batch_size", type=int, help="Name of task (dataset).")

    args = args.parse_args()

    # Load YAML file and update args
    with open("config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    merge(args, config_data['folders'])
    merge(args, config_data['run_pipeline'])


    # assemble the paths
    dataset_path = os.path.join(os.getcwd(), args.dataset_folder, args.task)
    weights_path = os.path.join(os.getcwd(), args.weights_folder)
    segment_path = os.path.join(dataset_path, args.segment_folder)
    detect_path = os.path.join(dataset_path, args.detect_folder)
    heatmap_path = os.path.join(dataset_path, args.heatmap_folder)

    if args.train:
        train(
            task_name=args.task,
            weights_path=weights_path,
            dataset_path=dataset_path,
            train_folder=args.train_folder,
            eval_folder=args.eval_folder,
            batch_size=args.batch_size,
        )

    if args.test:
        test_folder = os.path.join(dataset_path, args.test_folder)
        predictions = test(args.task, weights_path, test_folder)

    if args.segment or args.detect or args.heatmap:
        frames_folder = os.path.join(dataset_path, args.frame_folder)
        if not os.path.isdir(frames_folder):
            video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")
            extract_frames(video_path, frames_folder, frame_interval=1, save=True)

        process_frames(
            task_name=args.task,
            dataset_path=Path(frames_folder),
            weights_path=Path(weights_path),
            heatmap_path=Path(heatmap_path),
            segment_path=Path(segment_path),
            detect_path=Path(detect_path),
            segmentation=args.segment,
            detection=args.detect,
            heatmapping=args.heatmap,
            threshold=args.threshold
        )
