import argparse
from src.patchcore import test, train, process_frames
from src.data_collection import extract_frames
import matplotlib
matplotlib.use('TkAgg')  # This will force matplotlib to use an interactive backend
import os


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", required=True, type=str, help="Name of task (dataset).")

    #ACTIONS
    args.add_argument('--train', action='store_true', help="Training required.")
    args.add_argument('--test', action='store_true', help="Testing required.")
    args.add_argument('--segment', action='store_true', help="Segmentation required.")
    args.add_argument('--detect', action='store_true', help="Classification required.")
    args.add_argument('--heatmap', action='store_true', help="Hitmapping required.")

    #HYPERPARAMS
    args.add_argument('--threshold', default=0.5, type=float, help="Plotting required.")
    args.add_argument("--batch_size", default=32, type=int, help="Name of task (dataset).")

    #FOLDERS
    args.add_argument("--weights_folder", default="weights", type=str, help="Folder containing weights.")
    args.add_argument("--dataset_folder", default="data", type=str, help="Folder containing weights.")
    args.add_argument("--test_folder", default="test", type=str, help="Folder containing weights.")
    args.add_argument("--train_folder", default="train/OK", type=str, help="Folder containing weights.")
    args.add_argument("--eval_folder", default="val/NG", type=str, help="Folder containing weights.")
    args.add_argument("--frame_folder", default="frames", type=str, help="Folder containing weights.")

    args = args.parse_args()

    dataset_path = os.path.join(os.getcwd(), args.dataset_folder, args.task)
    weights_path = os.path.join(os.getcwd(), args.weights_folder)

    if args.train:
        train(
            task_name=args.task,
            weights_path=weights_path,
            dataset_path=dataset_path,
            train_folder=f"{args.train_folder}",
            eval_folder=f"{args.eval_folder}",
            batch_size=args.batch_size,
        )

    if args.test:
        test_folder = os.path.join(dataset_path, args.test_folder)
        predictions = test(args.task, weights_path, test_folder)

    if args.segment or args.detect or args.heatmap:
        video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")
        frames_folder = os.path.join(dataset_path, args.frame_folder)
        if not os.path.isdir(frames_folder):
            extract_frames(video_path, frames_folder, frame_interval=1)
        predictions = process_frames(
            task_name=args.task,
            dataset_path=frames_folder,
            weights_path=weights_path,
            heatmap_path=os.path.join(dataset_path, 'heatmap'),
            segment_path=os.path.join(dataset_path, 'segment'),
            detect_path=os.path.join(dataset_path, 'detect'),
            segmentation=args.segment,
            detection=args.detect,
            heatmapping=args.heatmap,
            threshold=args.threshold
        )
