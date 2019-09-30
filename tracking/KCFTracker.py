import cv2

class Tracker():

    def __init__(self, img, roi):
        self.target_type = 'Unknown'
        self.kcf = cv2.TrackerKCF_create()
        self.kcf.init(img, roi)
        self.roi = roi
        self.confidence = 1.0
        self.last_image = img

    def update(self, img, global_param=None):
        ret, new_roi = self.kcf.update(img)

        if ret :
            new_roi = [int(i) for i in new_roi]

            if self.target_type == '':

                disp = self.last_image.copy()
                cv2.rectangle(disp, (self.roi[0], self.roi[1]), (self.roi[0] + self.roi[2], self.roi[1] + self.roi[3]),
                              (0, 255, 0))
                cv2.imshow('Before', disp)

                disp = img.copy()
                cv2.rectangle(disp, (new_roi[0], new_roi[1]), (new_roi[0] + new_roi[2], new_roi[1] + new_roi[3]),
                              (0, 255, 0))
                cv2.imshow('After', disp)
                cv2.waitKey()

            self.roi = new_roi
            return self.roi

        return None