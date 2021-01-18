# Setting up GPU capability on Windows and Linux (January 2021)

Setting up the GPU capability on Windows or Linux for deep learning (using TensorFlow or PyTorch) is often not very obvious.

Following the documentation created by NVIDIA (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) is not obvious and I have not managed to make it work directly from there.

Instead, I took inspiration from [here](https://medium.com/swlh/cuda-installation-in-windows-2020-638b008b4639) and [here](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781).

The following will allow for generic installation of the required drivers and packages for TensorFlow, as PyTorch tends to included CuDNN in the package installation available in Conda.

## Generic requirements

For installing TensorFlow with GPU, you can check the requirements on [tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu). It is important to check the tested build configurations for [Windows](https://www.tensorflow.org/install/source_windows) and [Linux](https://www.tensorflow.org/install/source#linux) since this gives the compatibility matrix of a TensorFlow version with the respective CUDA and CuDNN versions.

Compatibility matrix for Windows:

![Compatibility matrix for Windows][comp_linux]

[comp_linux]: images/compatibility_windows.png "Compatibility matrix for Windows"

Compatibility matrix for Linux:

![Compatibility matrix for Linux][comp_linux]

[comp_linux]: images/compatibility_linux.png "Compatibility matrix for Linux"

## Installation on Windows

Work in progress

## Installation on Linux (Ubuntu 20.04)

Work in progress