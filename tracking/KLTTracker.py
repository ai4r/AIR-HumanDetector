import cv2
import numpy as np

class Tracker ():

    def __init__(self, img, roi):
        self.roi = roi
        self.dx = 0
        self.dy = 0

        self.feature_params = dict(maxCorners=100, qualityLevel=0.5, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(64, 64), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        roi_mask = np.zeros(shape=(img.shape[0], img.shape[1], 1), dtype=np.uint8)
        roi_mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]] = 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.pts = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **self.feature_params)

        if self.pts is not None:
            self.pts_count = len(self.pts)
            self.old_gray = gray

    def update(self, img, global_param=None):
        if self.pts is None:
            return None

        new_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, self.pts, None, **self.lk_params)

        good_new = new_pts[st == 1]

        x_sum = 0
        y_sum = 0
        for pt in self.pts:
            x_sum += pt[0][0]
            y_sum += pt[0][1]

        old_center = (int(x_sum / len(self.pts)), int(y_sum / len(self.pts)))

        x_sum = 0
        y_sum = 0
        for pt in new_pts:
            x_sum += pt[0][0]
            y_sum += pt[0][1]

        new_center = (int(x_sum / len(new_pts)), int(y_sum / len(new_pts)))

        if False:
            disp = self.old_gray.copy()
            for pt in self.pts:
                cv2.circle(disp, (pt[0][0], pt[0][1]), 2, (255, 255, 255))
            cv2.imshow('Old', disp)

            disp = new_gray.copy()
            for pt in new_pts:
                cv2.circle(disp, (pt[0][0], pt[0][1]), 2, (255, 255, 255))
            cv2.imshow('New', disp)
            cv2.waitKey()

        self.pts = new_pts
        self.old_gray = new_gray.copy()

        if len(good_new) <= 0:
            return None
        else:
            self.dx = new_center[0] - old_center[0]
            self.dy = new_center[1] - old_center[1]

            x = self.roi[0] + self.dx
            y = self.roi[1] + self.dy
            w = self.roi[2]
            h = self.roi[3]

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > img.shape[1]:
                w = img.shape[1] - x
            if y + h > img.shape[0]:
                h = img.shape[0] - y

            self.roi = (x, y, w, h)
            self.confidence = self.pts_count / len(good_new)
            return self.roi