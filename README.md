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
pip install -r requirements.txt
```

## Usage

### Train a Model

```bash
python train.py --config config.yaml
```

Customize `config.yaml` to modify model parameters.

### Test the Model

```bash
python test.py --checkpoint path/to/checkpoint.pth
```

### Generate Heatmaps & Segmentation Maps

```bash
python extract.py --image path/to/image.jpg
```

## Dependencies

- Python 3.10
- Ubuntu 20.04
- [Anomalib](https://github.com/openvinotoolkit/anomalib)
- OpenCV, PyTorch, NumPy