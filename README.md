# Anomaly Detection using PathCore (Anomalib)

This repository implements **anomaly detection** using **PathCore** from the [Anomalib](https://github.com/openvinotoolkit/anomalib) library. It supports **training, testing, and extracting** various outputs such as:

- Heatmaps
- Segmentation maps
- Detection results

## Features

- Train anomaly detection models  
- Test on new data  
- Extract heatmaps & segmentation maps  
- Perform object detection  

## Installation

### Set Up Environment

Tested with **Python 3.10** on **Ubuntu 20.04**.

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -e .
```
## Setup Dataset

### General Structure
Place data in the location specified in the file config.yaml under folders.dataset_folder. Make sure that it follows the following data tree:
```bash
|-- name
|-----|----- train
|-----|--------|------ OK
|-----|----- test
|-----|--------|------ OK
|-----|--------|------ NG
|-----|----- val
|-----|--------|------ OK
|-----|--------|------ NG
```

### Extract a dataset from video
A dataset can be extracted from a video. To do so, the video must be placed in the folder videos as `videoname.mp4`.
Then follow the following commands to extract `n` samples:
```bash
pip install -e .
```

## Usage

### Collect dataset

```bash
    python3 collect_dataset --task=videoname --n_samples=n
```
where the flag `--from_semiframes` can be used when not enough frames would be "anomaly-free"

### Train a Model

```bash
python run_pipeline.py --train --task=dataname
```
where `dataname` is the name of the dataset.
Customize `config.yaml` to modify model parameters.

### Test the Model

```bash
python run_pipeline.py --test --task=dataname
```

### Generate Heatmaps & Segmentation Maps

```bash
python run_pipeline.py --segement --detect --heatmap
```
where the flags `--segement`, `--detect` and `--heatmap` can be deactivated or deactivated whenever segmentation, detection or heatmapping is needed.

## Dependencies

- Python 3.10
- Ubuntu 20.04
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- OpenCV, PyTorch, NumPy