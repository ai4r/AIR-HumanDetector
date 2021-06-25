## Intro

KAIST Human Detection Tracking Module for HumanCare Project

Originally forked from DarkFlow (https://github.com/thtrieu/darkflow)

Open VINO reidentification from Intel Open VINO(http://docs.openvinotoolkit.org/latest/index.html)

## Demo

<https://youtu.be/ZLUshr_FVL8>

## Environment

Test on Ubuntu 18.04, Python3.6.9
## 3rd Party
openvino 2021.3
CUDA 10.0
CUDNN > 7
## Dependencies
python dependencies

opencv-python> 4.2.0.32

scipy>1.2.0

tensorflow-gpu==1.14.0

tensorlayer==1.11.1

cython 0.29.21


## Getting Started

1. Download the bin and ckpt folder which contains weight files from [HERE](https://drive.google.com/drive/folders/1MrRMU1dVP_WLaEqxMGfhB5HPeBwA22Ac?usp=sharing) and place it at base directory.

2. Install the OpenVino 2021.3

http://docs.openvinotoolkit.org/latest/index.html

3. Make the symbolic link for openvino
   ```
   ln -s <repository_base_path>/openvino/openvino_lib <openvino_install_path>/python/python3.6

   ```
3. Install the darkflow
    ```
    python3 setup.py build_ext --inplace
    ```

5. Install the dependencies
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Demo on sample video for the new model
    ```
    python Demo_vino.py
    ```

Press q button to quit the program.
Press c button to stop the program, press any key to continue.