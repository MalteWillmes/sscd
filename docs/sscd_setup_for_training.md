# SSCD Training - Setting up

The Salmon Scale Circuli Detector (SSCD) consists of two separate object detection models: the focus detector and the circulus detector. Each detector is a YOLOv3 (*You Only Look Once*), a Convolutional Neural Network (CNN), trained for its specific purpose.

Evidence of deterioration in SSCD's performance on new images will warrant the need for retraining one of the detectors (or both).

> **Note**: the step-by-step training protocol (previously the Jupyter notebook `docs/SSCD Training Protocol.ipynb`) has been removed from this fork and remains available in the git history. The training code path has not yet been updated for TensorFlow 2.16+/Keras 3.

Additional software requirements for training purposes depend on whether GPU acceleration is available, which would be the desirable hardware setting.

> For CPU-only processing, simply follow the standard installation in the [README][1]. No need for additional steps!

Next we describe how to set-up a dedicated Tensorflow-GPU workstation.


## Setting up Tensorflow with GPU support

> **Note**:
>
> *On setting up Tensorflow-GPU for the first time, configuring GPU drivers and libraries (CUDA, cuDNN) can be complex as component versions must be compatible. Refer to [TensorFlow's GPU installation guide][4] for detailed instructions.*
>
> *Using UV, GPU setup is straightforward: install the standard environment and configure the GPU drivers and libraries as described below.*
>
> *TensorFlow does not support GPUs on native Windows (TensorFlow >= 2.11); use Linux or WSL2.*

TensorFlow 2.16 and later includes GPU support in the main `tensorflow` package — there is no separate `tensorflow-gpu` package. Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd sscd
uv sync --dev
```


[1]: ../README.md
[4]: https://www.tensorflow.org/install/gpu
