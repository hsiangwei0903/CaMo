# Camera Motion Grounded Evaluation and Training for Vision-Language Models

This repository contains the code for paper Towards Genuine Spatial Intelligence: Camera Motion Grounded Evaluation and Training for Vision-Language Models.

## 🚀 Quick Links

- **Model Checkpoint**: [CaMo-3B](https://huggingface.co/hsiangwei0903/CaMo-3B)
- **Dataset**: [SpatialLadder-26k](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k) [CameraBench](https://docs.google.com/forms/d/e/1FAIpQLScQtbcgCM9aDLR5jhSm_mIPY3UL0Zqbpt18ijylf2DR1JnoFg/viewform)

## 📦 Installation

### Create Conda Environment

First, create a conda environment with Python 3.10:

```bash
conda create -n camo python=3.10 -y
conda activate camo
```

### Install LLaMA-Factory

Follow the standard LLaMA-Factory installation instructions:

```bash
# Clone the repository
git clone https://github.com/hsiangwei0903/CaMo.git

# Install dependencies
pip install -e .
pip install -r requirements/metrics.txt
```

Optional dependencies available: `metrics`, `deepspeed`. Install with: `pip install -e . && pip install -r requirements/metrics.txt -r requirements/deepspeed.txt`

For detailed setup instructions and other installation issues, refer to the [LLaMA-Factory documentation](https://github.com/hiyouga/LLaMA-Factory).

## 📚 Dataset Preparation

### Requirements

1. **SpatialLadder-26k**: Download from [Hugging Face](https://huggingface.co/datasets/hongxingli/SpatialLadder-26k)
2. **CameraBench**: Request access using [this form](https://docs.google.com/forms/d/e/1FAIpQLScQtbcgCM9aDLR5jhSm_mIPY3UL0Zqbpt18ijylf2DR1JnoFg/viewform)

### Setup

After downloading, organize the datasets in the `data/` directory and update the configuration files accordingly.

## 🏋️ Training

### Quick Start

To train the model, run the provided SLURM script:

```bash
sbatch train.slurm
```

### Alternative: Direct Command

Alternatively, you can run the LLaMA-Factory training command directly:

```bash
llamafactory-cli train examples/train_full/qwen2_5vl_3b_full_sft_camo.yaml
```

For more training configurations, refer to `train.slurm` or explore the `examples/train_full/` directory.

## 📊 Evaluation

### Supported Datasets

The evaluation pipeline supports the following spatial understanding benchmarks:

- **[VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench)**: Visual Spatial Intelligence Benchmark
- **[SPBench](https://huggingface.co/datasets/hongxingli/SPBench)**: Spatial Perception Benchmark
- **[CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)**: Computer Vision Benchmark
- **[SPAR-Bench](https://huggingface.co/datasets/jasonzhango/SPAR-Bench)**: Spatial Reasoning Benchmark
- **[ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench)**: View-aware Spatial Benchmark

### Setup

Before running evaluations, download the required benchmark datasets and update their paths in `eval_spld/evaluator.py`.

### Quick Start

To evaluate your trained model:

```bash
cd eval_spld
bash run_eval.sh
```

This command will execute the full evaluation pipeline using the default configuration.

### Spatial Narrative Score (SNS) Evaluation

#### Setup

First, export your Gemini API key as an environment variable:

```bash
export gemini_api_key=<your_gemini_api_key>
```

#### Running Evaluation

To evaluate video captions using the Spatial Narrative Score metric:

```bash
python caption_eval/eval_sns.py --results_path <caption_results_path>
```

The results JSON file should follow this format:

```json
{
    "47334107.mp4": [
        "Caption segment 1",
        "Caption segment 2",
        "Caption segment 3"
    ],
    "another_video.mp4": [
        "Caption segment 1",
        "Caption segment 2"
    ]
}
```

Each video file is mapped to a list of caption strings representing temporal segments of the video.

## 🔗 Related Resources

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): The underlying training framework
- [SpatialLadder](https://github.com/ZJU-REAL/SpatialLadder): Spatial understanding dataset and evaluation
