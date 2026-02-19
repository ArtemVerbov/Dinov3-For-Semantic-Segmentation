# Semantic Segmentation with Dinov3 Backbone

<p align="center">
  <img src="media/0_img.jpeg" alt="Cover Image" width="480" height="640"/>
</p>

This project provides a template for training a semantic segmentation model using a Feature Pyramid Network (FPN) architecture with a powerful Dinov3 backbone.

## Features

*   **Dinov3 Backbone:** Leverages the power of the latest self-supervised models from Meta AI.
*   **FPN Architecture:** Efficiently combines features from different scales for accurate segmentation.
*   **ClearML Integration:** Track your experiments and manage your models with ClearML (optional).
*   **Customizable:** Easily adapt the code to train on your own dataset.

## Results

Model was evaluated on Oxford-IIIT Pet Dataset and archives following results:

| Metric     | Score  |
|------------|--------|
| Dice Score | 0.8858 |
| mIoU       | 0.7568 |

## Prerequisites
*   PyTorch 2.1.0+
*   CUDA 12.1+ (for GPU training)
*   Access to the Dinov3 model on Hugging Face (see installation steps)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ArtemVerbov/Dinov3-For-Semantic-Segmentation.git
    cd Dinov3-For-Semantic-Segmentation
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements/requirments-dev.txt
    ```

3.  **Request Model Access:**
    The Dinov3 model requires you to request access on Hugging Face.
    *   Go to the [Dinov3 model page](https://huggingface.co/facebook/dinov3-large).
    *   Fill out the form to request access.
    *   Log in to your Hugging Face account in your terminal [Command Line Interface](https://huggingface.co/docs/huggingface_hub/en/guides/cli)

## Configuration

This project uses a modular configuration system managed by Hydra. The parameters are organized into several YAML files located in the `configs/` directory. The main entry point is `train.yaml`.

*   `configs/train.yaml`: The main configuration file. It defines the training loop settings (epochs, precision, logging), optimizer, learning rate scheduler, and composes the other configuration files.
*   `configs/model.yaml`: Defines the model architecture. Here you can set the Dinov3 backbone variant, FPN parameters, number of output classes.
*   `configs/data.yaml`: Specifies all data-related parameters, including the dataset name, image size, batch size, and data loader settings.
*   `configs/project.yaml`: Handles project-level settings, such as the project and experiment names used for logging (e.g., in ClearML).
## Usage

### Training

To start training the model, run the following command:

```bash
python -m src.train