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

class TrackMod_own():
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD, feature_type='yolo', classifier_type='KNC', cropping_ratio=(0.8,0.8)):

        self.name = ''
        self.conf_path = conf_file
        self.model_path = model_file
        self.det_threshold = det_threshold
        self.min_det_threshold = min_det_treshold
        self.feature_grid = [4, 4, -1]
        self.current_tracks = []
        self.cropping_ratio = cropping_ratio


        self.featureType = feature_type
        self.tracking = Tracking_own.Tracking_own(featureType=self.featureType, classifierType=classifier_type)

        # Setting for Detection
        self.det_net = None
        options = {"model": conf_file, "load": model_file, "threshold": self.min_det_threshold, 'gpu': 1.0}

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