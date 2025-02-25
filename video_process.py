import cv2
import os
import argparse


def assemble_video(frames_path, output_path, fps=30):

    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith((".png", ".jpg", ".jpeg"))])

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release resources
    video.release()
    print(f"Video saved as {output_path}")


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--task", required=True, type=str, help="Name of the task")
    args.add_argument("--task_folder", required=True, type=str, help="Name of task folder (dataset).")

    args.add_argument("--fps", type=int, default=30, help="Frames per second for video.")

    args.add_argument("--dataset_folder", default="data", type=str, help="Folder containing datasets.")
    args = args.parse_args()

    data_path = os.path.join(os.getcwd(), args.dataset_folder, args.task_folder, args.task)
    output_path = f"videos/results"
    os.makedirs(output_path, exist_ok=True)
    assemble_video(
        frames_path=data_path,
        output_path=os.path.join(output_path, f"{args.task_folder}_{args.task}_{args.fps}fps.mp4"),
        fps=args.fps
    )