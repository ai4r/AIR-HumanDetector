## Intro

KAIST Human Detection Tracking Module for HumanCare Project
Originally forked from DarkFlow (https://github.com/thtrieu/darkflow)

## Demo

<https://youtu.be/ZLUshr_FVL8>

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

## Getting Started

1. Download weight file from <https://drive.google.com/drive/folders/1a8n649fCmbumeyoBIAU9nh1kIFnt6abp?usp=sharing>
	
2. Put /bin folder at base directory
	
3. Build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

4. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

5. Or install with pip globally
    ```
    pip install .
    ```

## Usage

```
init(conf_file="cfg/yolo-f.cfg", model_file=8000)

USE_CMATCH = True
while True:
	img = cv2.imread('test.txt')

	if img is None:
	    break

	run(img)

	for tr in current_tracks:
	    cv2.rectangle(img, tr.tl, tr.br, (0, 255, 0), 1)

	cv2.imshow('Result', img)
	key = cv2.waitKey(20)
	if key == ord('q') :
	    break
```

'img' must be ndarray type

## LICENSE

This software is a part of AIR, and follows the [AIR License and Service Agreement]<https://github.com/ai4r/AIR-Act2Act/blob/master/LICENSE.md>.
