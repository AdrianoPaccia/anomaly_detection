import argparse
import os
from anomaly_detection.data_collection import extract_frames
from anomaly_detection.utils import dataset_split_and_save
from anomaly_detection.utils import merge
import yaml

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--task", required=True, type=str, help="Name of task (dataset).")
    args.add_argument('--from_semiframes', action='store_true',
                      help="Collect the frames for training by assembling semiframes.")
    args.add_argument("--frame_interval", type=int, help="sampling interval between frames.")
    args.add_argument("--n_samples", type=int, help="Number of samples to assembele.")
    args = args.parse_args()


    # Load YAML file and update args
    with open("config.yaml", "r") as file:
        config_data = yaml.safe_load(file)
    merge(args, config_data['folders'])
    merge(args, config_data['collect_dataset'])

    data_path = os.path.join(os.getcwd(), "data", args.task)
    video_path = os.path.join(os.getcwd(), "videos", f"{args.task}.mp4")

    if args.from_semiframes:
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
        output_path = os.path.join(os.getcwd(), "data", args.task, f"train/OK")
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
        split_ratio=args.split_ratio
    )