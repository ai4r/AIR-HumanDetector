'''
Human Detector - Test code
'''

import os
import time
import sys
import cv2
import DataProvider
import numpy as np
from calcHistogram import calcHistFeature

import tracking.Tracking as Tracking

from darkflow.net.build import TFNet


event_log = []
online_pos_images = []
online_neg_images = []

FEATURETYPE = 'colorHist'
#FEATURETYPE = 'yolo'
# feature type: yolo, colorHist

colorBins = 16

class TrackMod:
    def __init__(self, conf_file, model_file, det_threshold=0.5, min_det_treshold=0.5, tracker_type='',
                 tracker_limit=60, use_cmatch=False):

        self.name = ''
        self.conf_path = conf_file
        self.model_path = model_file
        self.det_threshold = det_threshold
        self.min_det_threshold = min_det_treshold
        self.use_cmatch = use_cmatch
        self.feature_grid = [4, 4, -1]
        self.current_tracks = []
        self.show_online_data = False

        # feature type: yolo, colorHist, streetCosavelor, streetPattern, streetAuto
        # self.featureType = 'colorHist'
        self.featureType = FEATURETYPE

        self.tracking = Tracking.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                          tracker_limit=tracker_limit, featureType=self.featureType)

        # Setting for Detection
        self.det_net = None
        options = {"model": conf_file, "load": model_file, "threshold": self.min_det_threshold, 'gpu': 2.0}
        self.det_net = TFNet(options)



        self.tracking.feature_func = self.get_feature

        # Setting for Tracking
    def get_feature(self, img, rectList):
        global colorBins
        if self.featureType == 'yolo':
            f = self.det_net.return_features(rectList, self.feature_grid)
        elif self.featureType == 'colorHist':
            assert colorBins != 0, 'ColorBins is zero...'

            f = calcHistFeature(img, rectList, colorBins)
        elif self.featureType == 'streetColor':
            f = None
        return f


    def run(self, image):
        result = self.det_net.return_predict(image)

        new_tracks = []
        new_tracks_rect = []

        candidate_tracks = []
        candidate_tracks_rect = []

        for r in result:
            class_name = r['label']
            conf = r['confidence']

            if not class_name in ['personTLD', 'person', 'body', 'face']:
                continue

            if class_name == 'personTLD':
                class_name = 'body'

            t = Tracking.Track(class_name, r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'],
                               r['bottomright']['y'], conf, image, featureType=self.featureType)

            # High-Confidence Tracks
            if t.type == 'body':
                if conf >= self.det_threshold:
                    new_tracks.append(t)
                    new_tracks_rect.append(t.rect)
                # Low-Confidence Tracks
                elif conf >= self.min_det_threshold:
                    candidate_tracks.append(t)
                    candidate_tracks_rect.append(t.rect)
            elif t.type == 'face':
                if conf >= self.det_threshold * 1:
                    new_tracks.append(t)
                    new_tracks_rect.append(t.rect)
                # Low-Confidence Tracks
                elif conf >= self.min_det_threshold:
                    candidate_tracks.append(t)
                    candidate_tracks_rect.append(t.rect)

        # Setting for Candidate Match
        #new_tracks_features = self.det_net.return_features(new_tracks_rect, self.feature_grid)

        new_tracks_features = self.get_feature(image, new_tracks_rect)

        for idx, tr in enumerate(new_tracks):
            new_tracks[idx].feature = new_tracks_features[idx]

        #candidate_tracks_features = self.det_net.return_features(candidate_tracks_rect, self.feature_grid)

        candidate_tracks_features = self.get_feature(image, candidate_tracks_rect)

        for idx, tr in enumerate(candidate_tracks):
            candidate_tracks[idx].feature = candidate_tracks_features[idx]

            # Add negative samples for online learning to event images
            if self.show_online_data:
                sub_img = image[tr.tl[1]:tr.br[1], tr.tl[0]:tr.br[0], :].copy()
                online_neg_images.append(sub_img)

        while len(online_neg_images) > 17:
            online_neg_images.remove(online_neg_images[0])

        self.tracking.candidate_tracks = candidate_tracks
        self.tracking.candidate_features = candidate_tracks_features

        # Update
        self.tracking.update(self.current_tracks, new_tracks, image)

        # Add positive samples for online learning to event images
        if self.show_online_data:
            for tr in self.current_tracks:
                sub_img = image[tr.tl[1]:tr.br[1], tr.tl[0]:tr.br[0], :].copy()
                online_pos_images.append(sub_img)
            while len(online_pos_images) > 17:
                online_pos_images.remove(online_pos_images[0])

    def draw(self, img):
        for tr in self.current_tracks:
            c = (0, 255, 0)
            if tr.last_state == 'Search':
                c = (255, 0, 0)
            elif tr.last_state == 'CMatch':
                c = (0, 255, 255)


            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def close(self):
        self.det_net = None


def test():

    # DEBUG is True for showing frame, rectangle and frame ID
    DEBUG=True
    # set the test video
    fileList = ['test.mp4']

    for eachFile in fileList:
        dp = DataProvider.VideoDataProvider(eachFile)

        # Resize the frame
        resize_rate = 0.4

        res = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=0.2, min_det_treshold=0.1,
                      use_cmatch=True, tracker_type='TM', tracker_limit=10)

        frameIdx = 0
        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            # running tracking process
            res.run(img_resize)

            bboxListT2 = []

            print("For Frame " + str(frameIdx))

            for trackRes in res.current_tracks:
                id = trackRes.id
                x1 = trackRes.tl[0]
                y1 = trackRes.tl[1]
                x2 = trackRes.br[0]
                y2 = trackRes.br[1]
                last_state = trackRes.last_state

                type = trackRes.type

                bbox = (x1, y1, x2, y2)

                if(DEBUG):
                    cv2.putText(img_resize, str(id), (x1+20, y1+40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
                    cv2.rectangle(img_resize, (x1, y1), (x2, y2), (0,0,255), 2)


                print('ID: %d, bbox(x1,y1,x2,y2): (%d %d %d %d)'%(id, x1, y1, x2, y2))


            frameIdx += 1

            if(len(res.current_tracks) > 0 and DEBUG):
                cv2.imshow('currentFrame', img_resize)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.exit(test())

