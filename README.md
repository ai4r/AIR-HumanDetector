## Intro

KAIST Human Detection Tracking Module for HumanCare Project
Originally forked from DarkFlow (https://github.com/thtrieu/darkflow)

## Demo

<https://youtu.be/ZLUshr_FVL8>

## Environment

Test on Ubuntu 16.04, Python3.5.2, Python 2.7.12

## Dependencies

opencv-python> 4.2.0.32

scipy>1.2.0

tensorflow-gpu==1.12.0

tensorlayer==1.11.0

cython==0.29.21

## Getting Started

1. Download the bin and ckpt folder which contains weight files from [HERE](https://drive.google.com/drive/folders/1MrRMU1dVP_WLaEqxMGfhB5HPeBwA22Ac?usp=sharing) and place it at base directory.

2. Install the darkflow
    ```
    python3 setup.py build_ext --inplace
    ```

3. Install the dependencies
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Demo video for comparing new and old models
    ```
    python Demo_dual.py
    ```
2. Demo video for the new model
    ```
    python Demo_new.py
    ```

3. Simple Test File
   ```
   python HumanDetector.py
   ```

Press q button to quit the program.
Press c button to stop the program, press any key to continue.



