## Intro

KAIST Human Detection Tracking Module for HumanCare Project
Originally forked from DarkFlow (https://github.com/thtrieu/darkflow)

## Demo

<https://youtu.be/ZLUshr_FVL8>

## Dependencies

opencv-python==4.2.0.34

scipy==1.1.0

tensorflow-gpu==1.12.0

tensorlayer==1.11.0

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

1. Demo video
    ```
    python Demo.py
    ```

2. Test with a given video
   ```
   python Demo.py --videoPath=VIDEOPATH
   ```

Press q button to quit the program

