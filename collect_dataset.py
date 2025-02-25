import argparse
import os
from src.data_collection import extract_semiframes, extract_frames, assemble_frames


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--extract_semiframes', action='store_true', help="Extract semiframes from the video.")
    args.add_argument('--extract_frames', action='store_true', help="Extract frames from the video.")
    args.add_argument('--assemble_dataset', action='store_true', help="Assemble semiframes.")
    args.add_argument("--task", required=True, type=str, help="Name of task (dataset).")
    args.add_argument("--frame_interval", type=int, default=5, help="sampling interval between frames.")
    args.add_argument("--n_samples", type=int, default=100, help="Number of samples to assembele.")
    args.add_argument("--resize", action='store_true', help="Number of samples to assembele.")

    args.add_argument("--weights_folder", default="weights", type=str, help="Folder containing weights.")
    args.add_argument("--dataset_folder", default="data", type=str, help="Folder containing weights.")
    args = args.parse_args()

    if args.extract_semiframes:
        video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")
        output_folder = os.path.join(os.getcwd(), "data", args.task, "semi_frames")
        extract_semiframes(video_path, output_folder, frame_interval=args.frame_interval)

    if args.extract_frames:
        video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")
        output_folder = os.path.join(os.getcwd(), "data", args.task, "test_resize" if args.resize else "test")
        extract_frames(video_path, output_folder, frame_interval=5, resize=args.resize)

    if args.assemble_dataset:
        semiframes_path = os.path.join(os.getcwd(), "data", args.task, "semi_frames")
        output_folder = os.path.join(os.getcwd(), "data", args.task, f"train{'_resize' if args.resize else ''}/OK")
        assemble_frames(args.n_samples, semiframes_path, output_folder, resize=args.resize)