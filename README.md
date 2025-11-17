# MultiModal Material Estimation

A multimodal material estimation project that utilizes both audio and visual information for material classification.

## Quick Start

### 1. Pull Docker Image

Pull the Docker image from Docker Hub:

```bash
docker pull timttu/multimodal-material-estimation:latest
```

Alternatively, you can download the image from the Docker Hub repository (see links below).

### 2. Run Docker Container

After pulling the image, run the container with GPU support:

```bash
docker run -it --gpus all timttu/multimodal-material-estimation:latest /bin/bash
```

### 3. Using Checkpoints

Once inside the Docker container, you can choose from different checkpoint options:

#### Option A: Original Training Checkpoint

If you want to use the original checkpoint from the initial training, you can find it at the following path:

```
/MultiModalMaterialEstimation/ckpt.pth
```

This checkpoint achieves approximately **90%** accuracy.

#### Option B: Optimized Checkpoint (Recommended)

For better performance, you can use the optimized checkpoint that has been fine-tuned with different weight configurations:

```
/workspace/checkpoints/model_ckpt_finetune.pth
```

This checkpoint achieves approximately **92%** accuracy through weight optimization and fine-tuning.

## Usage

### Training

To train the model:

```bash
python train.py --config config.json
```

### Testing

To test the model with a specific checkpoint:

```bash
python test.py --config config_test.json --ckpt_path [checkpoint_path]
```

## Project Structure

- `train.py`: Model training script
- `test.py`: Model testing script
- `dataset_utils.py`: Dataset processing utilities
- `config.json`: Training configuration file
- `config_test.json`: Testing configuration file

## Dependencies

The project runs in a Docker container with all necessary dependencies pre-installed:

- PyTorch 1.10.0
- CUDA 11.3
- Transformers
- OpenAI Whisper
- CLIP
- Other related dependencies

## Requirements

- Docker installed on your system
- NVIDIA Docker runtime (for GPU support)
- Use the `--gpus all` flag when running the container to enable GPU support

## Links

1. **Docker Image Repository**: [Docker Hub - MultiModal Material Estimation](https://hub.docker.com/repository/docker/timttu/multimodal-material-estimation/general)

2. **Original Checkpoint Path**: `/MultiModalMaterialEstimation/ckpt.pth` (approximately 90% accuracy)

3. **Optimized Checkpoint Path**: `/workspace/checkpoints/model_ckpt_finetune.pth` (approximately 92% accuracy)
