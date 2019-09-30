## Intro

KAIST Human Detection Tracking Module for HumanCare Project
Originally forked from DarkFlow (https://github.com/thtrieu/darkflow)

## Demo

<https://youtu.be/ZLUshr_FVL8>

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

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
