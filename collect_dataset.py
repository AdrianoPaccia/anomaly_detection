import argparse
import os
from anomaly_detection.data_collection import extract_frames
from anomaly_detection.utils import dataset_split_and_save

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

    data_path = os.path.join(os.getcwd(), "data", args.task)
    video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")



    from_semiframes = True
    split_ratio = 0.8

    if from_semiframes:
        from anomaly_detection.data_collection import extract_empty_semiframes, assemble_semiframes

        #extract semiframes
        semiframes_path = os.path.join(data_path, "semi_frames")
        if not os.path.isdir(semiframes_path) or len(os.listdir(semiframes_path))==0:
            print('Extracting frames')
            extract_empty_semiframes(
                video_path=video_path,
                output_path=semiframes_path,
                empty_sample=os.path.join(data_path, "empty_sample.png"),
                frame_interval=args.frame_interval)

        #assemble frames
        print('Assembling frames')
        output_path = os.path.join(os.getcwd(), "data", args.task, f"train{'_resize' if args.resize else ''}/OK")
        assemble_semiframes(args.n_samples, semiframes_path, output_path)

    else:
        from anomaly_detection.data_collection import assemble_frames

        semiframes_path = os.path.join(os.getcwd(), "data", args.task, "semi_frames")
        output_path = os.path.join(os.getcwd(), "data", args.task, f"train/OK")
        assemble_frames(args.n_samples, semiframes_path, output_path, resize=args.resize)

    #collect and save also train and test dataset
    print('Assembling Test and Val datasets')
    output_path = os.path.join(os.getcwd(), "data", args.task, "test")
    frames = extract_frames(video_path, output_path, frame_interval=5, save=False)
    dataset_split_and_save(
        dataset=frames,
        path1=os.path.join(os.getcwd(), "data", args.task, "test"),
        path2=os.path.join(os.getcwd(), "data", args.task, "val"),
        split_ratio=split_ratio
    )