import cv2
import numpy as np
import elm.Utilities as util


class Tracker ():

    def __init__(self, img, roi, max_delta_x = 1.0, max_delta_y = 0.3, move_delta_x = 0.0, move_delta_y = 0.0,
                 image_type='HistEQ'):

        self.target_type = 'Unknown'
        self.roi = roi
        self.image = img
        self.target = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
        self.MaxDelta = (max_delta_x, max_delta_y)
        self.MoveDelta = (move_delta_x, move_delta_y)
        self.dx = 0
        self.dy = 0
        self.threshold = 0.9
        self.image_type = image_type

    def update(self, img, global_param=None):
        if len(self.target) == 0:
            return None

        x_max = int(self.roi[2] * self.MaxDelta[0])
        y_max = int(self.roi[3] * self.MaxDelta[1])

        x1 = self.roi[0] - x_max
        x2 = self.roi[0] + self.roi[2] + x_max
        y1 = self.roi[1] - y_max
        y2 = self.roi[1] + self.roi[3] + y_max

        if True:
            dx = int(self.dx * self.MoveDelta[0])
            x1 += dx
            x2 += dx

            dy = int(self.dy * self.MoveDelta[0])
            y1 += dy
            y2 += dy

        if x1 < 0:
            x1 = 0
        if x2 >= img.shape[1]:
            x2 = img.shape[1] - 1
        if y1 < 0:
            y1 = 0
        if y2 >= img.shape[0]:
            y2 = img.shape[0] - 1

        sub_img = img[y1:y2, x1:x2]

        def get_laplacian(src):
            dst = cv2.GaussianBlur(src, (3, 3), 0.1)
            dst = cv2.Laplacian(dst, cv2.CV_64F)
            dst = np.absolute(dst)
            dst = np.uint8(dst)
            return dst

        def get_sobel(src):
            dst = cv2.GaussianBlur(src, (3, 3), 0.1)
            sobelx = cv2.Sobel(dst, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(dst, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.uint8(sobelx + sobely)
            return sobel

        def get_histequalize(src):
            dst = cv2.GaussianBlur(src, (3, 3), 0.1)
            return util.equalizeHist(dst)

        if self.image_type == 'HistEQ':
            sub_img = get_histequalize(sub_img)
            target_img = get_histequalize(self.target)
        elif self.image_type == 'Lap':
            sub_img = get_laplacian(sub_img)
            target_img = get_laplacian(self.target)
        elif self.image_type == 'Sobel':
            sub_img = get_sobel(sub_img)
            target_img = get_sobel(self.target)
        else:
            raise Exception('Invalid image_type = ' + self.image_type)

        if sub_img.shape[0] < self.target.shape[0] or sub_img.shape[1] < self.target.shape[1]:
            return None

        res = cv2.matchTemplate(sub_img, target_img, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < self.threshold:
            return None
        else:
            new_roi = (x1 + max_loc[0], y1 + max_loc[1], self.roi[2], self.roi[3])
            self.dx = (new_roi[0] - self.roi[0])
            self.dy = (new_roi[1] - self.roi[1])
            self.roi = new_roi
            self.target = img[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]]
            self.confidence = 0.5

            if self.target_type == '':
                cv2.rectangle(sub_img, (max_loc[0], max_loc[1]), (max_loc[0] + self.roi[2], max_loc[0] + self.roi[3]),
                              (0, 0, 255))
                cv2.imshow('Search', sub_img)
                cv2.imshow('Target', target_img)
                cv2.imshow('Found', self.target)

            return self.roi
