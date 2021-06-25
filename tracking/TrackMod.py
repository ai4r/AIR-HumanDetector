import os
import time
import sys
import cv2
import DataProvider
import numpy as np

from calcHistogram import calcHistogram3D
from calcHistogram import calcOrigin

from openvino.vino_reidentification_module import get_reid_feature


import tracking.Tracking_old as Tracking_old
import tracking.Tracking as Tracking
import tracking.Tracking_own as Tracking_own

from darkflow.net.build import TFNet

DET_THRESHOLD=0.2
MIN_DET_THRESHOLD=0.1

class TrackMod:
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD, tracker_type='',
                 tracker_limit=60, use_cmatch=False, feature_type='yolo', classifier_type='KNC', cropping_ratio=(0.8,0.8)):

        self.name = ''
        self.conf_path = conf_file
        self.model_path = model_file
        self.det_threshold = det_threshold
        self.min_det_threshold = min_det_treshold
        self.use_cmatch = use_cmatch
        self.feature_grid = [4, 4, -1]
        self.current_tracks = []
        self.show_online_data = False
        self.cropping_ratio = cropping_ratio

        if feature_type == 'yolo':
            self.featureType = feature_type
            self.tracking = Tracking_old.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                              tracker_limit=tracker_limit, featureType=self.featureType)
        else:
            self.featureType = feature_type
            self.tracking = Tracking.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                              tracker_limit=tracker_limit, featureType=self.featureType,
                                              classifierType=classifier_type)



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
            f = calcHistogram3D(img, rectList, self.cropping_ratio)
        elif self.featureType == 'vino_reid':
            f = get_reid_feature(img, rectList, self.cropping_ratio)
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


        self.tracking.candidate_tracks = candidate_tracks
        self.tracking.candidate_features = candidate_tracks_features

        # Update
        self.tracking.update(self.current_tracks, new_tracks, image)


    def draw(self, img):
        for tr in self.current_tracks:
            c = (0, 255, 0)
            if tr.last_state == 'Search':
                c = (255, 0, 0)
            elif tr.last_state == 'CMatch':
                c = (0, 255, 255)

            cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0] + 20, tr.tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def close(self):
        self.det_net = None

class TrackMod_own(TrackMod):
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD, tracker_limit=60, use_cmatch=False, feature_type='yolo', classifier_type='KNC', cropping_ratio=(0.8,0.8)):

        self.name = ''
        self.conf_path = conf_file
        self.model_path = model_file
        self.det_threshold = det_threshold
        self.min_det_threshold = min_det_treshold
        self.use_cmatch = use_cmatch
        self.feature_grid = [4, 4, -1]
        self.current_tracks = []
        self.cropping_ratio = cropping_ratio


        self.featureType = feature_type
        self.tracking = Tracking_own.Tracking_own(featureType=self.featureType, classifierType=classifier_type)

        # Setting for Detection
        self.det_net = None
        options = {"model": conf_file, "load": model_file, "threshold": self.min_det_threshold, 'gpu': 2.0}

        self.det_net = TFNet(options)

        self.tracking.feature_func = self.get_feature

        # Setting for Tracking
    def get_feature(self, img, rectList):
        f = get_reid_feature(img, rectList, self.cropping_ratio)
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

        candidate_tracks_features = self.get_feature(image, candidate_tracks_rect)

        for idx, tr in enumerate(candidate_tracks):
            candidate_tracks[idx].feature = candidate_tracks_features[idx]


        self.tracking.candidate_tracks = candidate_tracks
        self.tracking.candidate_features = candidate_tracks_features

        # Update
        self.tracking.update(self.current_tracks, new_tracks, image)

    def close(self):
        self.det_net = None



class TrackMod_dual(TrackMod):
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD,
                 tracker_type='',
                 tracker_limit=60, use_cmatch=False, feature_type='yolo', classifier_type='KNC',
                 cropping_ratio=(0.8, 0.8)):
        self.track_list = ["KNN", "ELM"]
        self.name = ''
        self.conf_path = conf_file
        self.model_path = model_file
        self.det_threshold = det_threshold
        self.min_det_threshold = min_det_treshold
        self.use_cmatch = use_cmatch
        self.feature_grid = [4, 4, -1]
        self.current_tracks_old = []
        self.current_tracks_new = []
        self.show_online_data = False


        self.tracking_new = Tracking.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                                            tracker_limit=tracker_limit, featureType=feature_type,
                                                            classifierType=classifier_type)
        if self.tracking_new.featureType == "colorHist":
            self.tracking_new.feature_func = self.get_feature_color
        elif self.tracking_new.featureType == "origin":
            self.tracking_new.feature_func = self.get_feature_origin
        self.tracking_old = Tracking_old.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                                                tracker_limit=tracker_limit, featureType='yolo')
        self.tracking_old.feature_func = self.get_feature_yolo

        # Setting for Detection
        self.det_net = None
        options = {"model": conf_file, "load": model_file, "threshold": self.min_det_threshold, 'gpu': 2.0}
        self.det_net = TFNet(options)

    def get_feature_yolo(self,img, rectList):
        f = self.det_net.return_features(rectList, self.feature_grid)
        return f

    def run_old(self, image, result):
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

            t = Tracking_old.Track(class_name, r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'],
                               r['bottomright']['y'], conf, image, featureType=self.tracking_old.featureType)

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
        # new_tracks_features = self.det_net.return_features(new_tracks_rect, self.feature_grid)

        new_tracks_features = self.get_feature_yolo(image, new_tracks_rect)

        for idx, tr in enumerate(new_tracks):
            new_tracks[idx].feature = new_tracks_features[idx]

        candidate_tracks_features = self.get_feature_yolo(image, candidate_tracks_rect)

        for idx, tr in enumerate(candidate_tracks):
            candidate_tracks[idx].feature = candidate_tracks_features[idx]

        self.tracking_old.candidate_tracks = candidate_tracks
        self.tracking_old.candidate_features = candidate_tracks_features

        # Update
        self.tracking_old.update(self.current_tracks_old, new_tracks, image)
    def run_new(self, image, result):
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
                                   r['bottomright']['y'], conf, image, featureType=self.tracking_old.featureType)

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
        # new_tracks_features = self.det_net.return_features(new_tracks_rect, self.feature_grid)

        if self.tracking_new.featureType == "colorHist":
            new_tracks_features = self.get_feature_color(image, new_tracks_rect)
        elif self.tracking_new.featureType == "origin":
            new_tracks_features = self.get_feature_origin(image, new_tracks_rect)

        for idx, tr in enumerate(new_tracks):
            new_tracks[idx].feature = new_tracks_features[idx]
        if self.tracking_new.featureType == "colorHist":
            candidate_tracks_features = self.get_feature_color(image, candidate_tracks_rect)
        elif self.tracking_new.featureType == "origin":
            candidate_tracks_features = self.get_feature_origin(image, candidate_tracks_rect)

        for idx, tr in enumerate(candidate_tracks):
            candidate_tracks[idx].feature = candidate_tracks_features[idx]

        self.tracking_new.candidate_tracks = candidate_tracks
        self.tracking_new.candidate_features = candidate_tracks_features

        # Update
        self.tracking_new.update(self.current_tracks_new, new_tracks, image)

    def run(self, image):
        result = self.det_net.return_predict(image)
        self.run_old(image, result)
        self.run_new(image, result)

    def draw_old(self, img):
        for tr in self.current_tracks_old:
            c = (0, 255, 0)
            if tr.last_state == 'Search':
                c = (255, 0, 0)
            elif tr.last_state == 'CMatch':
                c = (0, 255, 255)

            cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0] + 20, tr.tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def draw_new(self, img):
        for tr in self.current_tracks_new:
            c = (0, 255, 0)
            if tr.last_state == 'Search':
                c = (255, 0, 0)
            elif tr.last_state == 'CMatch':
                c = (0, 255, 255)

            cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0] + 20, tr.tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def close(self):
        self.det_net = None

