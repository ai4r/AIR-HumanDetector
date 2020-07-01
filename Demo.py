import os
import time
import sys
import cv2
import DataProvider
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Test argument')
parser.add_argument('--videoPath', help='Video path')

args = parser.parse_args()

import tracking.Tracking as Tracking

from darkflow.net.build import TFNet

event_log = []
online_pos_images = []
online_neg_images = []

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

        self.tracking = Tracking.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                          tracker_limit=tracker_limit)

        # Setting for Detection
        self.det_net = None
        options = {"model": conf_file, "load": model_file, "threshold": self.min_det_threshold, 'gpu': 2.0}
        self.det_net = TFNet(options)

        # Setting for Tracking
        def get_feature(rect):
            f = self.det_net.return_features([rect], self.feature_grid)
            return f

        self.tracking.feature_func = get_feature

    def run(self, image):
        result = self.det_net.return_predict(image)

        new_tracks = []
        new_tracks_rect = []

        candidate_tracks = []
        candidate_tracks_rect = []

        for r in result:
            class_name = r['label']
            conf = r['confidence']

            if not class_name in ['person', 'body', 'face']:
                continue

            if class_name == 'person':
                class_name = 'body'

            t = Tracking.Track(class_name, r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'],
                               r['bottomright']['y'], conf, image)

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
        new_tracks_features = self.det_net.return_features(new_tracks_rect, self.feature_grid)
        for idx, tr in enumerate(new_tracks):
            new_tracks[idx].feature = new_tracks_features[idx]

        candidate_tracks_features = self.det_net.return_features(candidate_tracks_rect, self.feature_grid)
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

            cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0] + 20, tr.tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def close(self):
        self.det_net = None

def event_handler(sender, type, data):
    if type == 'Message':
        current_time = time.strftime("%H:%M:%S", time.localtime())
        event_log.append('[%s] %s' % (current_time, data))

    if len(event_log) > 5:
        event_log.remove(event_log[0])

def draw_panel(img1, img2):
    panel_img = np.zeros(shape=(720, 1080, 3), dtype=np.uint8)

    h1, w1, c1 = img1.shape
    panel_img[20:20+h1, 20:20+w1, :] = img1

    h2, w2, c2 = img2.shape
    panel_img[20:20+h2, 40+w1:40+w1+w2, :] = img2

    sub_img_top = max(h1, h2) + 50
    sub_img_size = (64, 64)

    cv2.rectangle(panel_img, (20, sub_img_top), (1060, sub_img_top + 135), (255, 255, 255), 1)
    for img_idx, data in enumerate(online_pos_images):
        x_offset = 30 + (img_idx * 60)
        img = cv2.resize(data, sub_img_size)
        panel_img[sub_img_top + 5: sub_img_top + sub_img_size[0] + 5, x_offset:x_offset + sub_img_size[0], :] = img

    sub_img_top += sub_img_size[1]
    for img_idx, data in enumerate(online_neg_images):
        x_offset = 30 + (img_idx * 60)
        img = cv2.resize(data, sub_img_size)
        panel_img[sub_img_top + 5: sub_img_top + sub_img_size[0] + 5, x_offset:x_offset + sub_img_size[0], :] = img

    log_top = sub_img_top + 100
    cv2.rectangle(panel_img, (20, log_top), (1060, log_top + 120), (255, 255, 255), 1)

    for msg_idx, msg in enumerate(event_log):
        y_offset = log_top + 20 + (msg_idx * 20)
        cv2.putText(panel_img, msg, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return panel_img

def main(argv=None):

    if(args.videoPath is None):
        print('No external video path found...')
        print('Instead, we use testVideo...')
        dp = DataProvider.VideoDataProvider('data/Tracker(2atthesametime).mp4')
    else:
        if (not os.path.exists(args.videoPath)):
            print('No video path ' + args.videoPath + ' not exists...')
            return
        dp = DataProvider.VideoDataProvider(args.videoPath)


    resize_rate = 0.4

    if(not os.path.exists('cfg/yolo.cfg')):
        print('cfg/yolo.cfg not exists...')
        return
    if (not os.path.exists('bin/yolo.weights')):
        print('bin/yolo.weights not exists...')
        return
    if (not os.path.exists('cfg/yolo-f.cfg')):
        print('cfg/yolo-f.cfg not exists...')
        return

    t1 = TrackMod(conf_file="cfg/yolo.cfg", model_file='bin/yolo.weights',
                  det_threshold=0.5, min_det_treshold=0.5,
                  use_cmatch=False, tracker_type='TM', tracker_limit=60)

    t2 = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000,
                  det_threshold=0.2, min_det_treshold=0.1,
                  use_cmatch=True, tracker_type='TM', tracker_limit=10)

    t2.tracking.event_func = event_handler
    t2.show_online_data = True

    while True:
        img = dp.get()
        if img is None:
            break

        img = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

        t1.run(img)
        t2.run(img)

        img2 = img.copy()
        t1.draw(img)
        t2.draw(img2)

        panel = draw_panel(img, img2)

        cv2.imshow('Tracking Result', panel)
        key = cv2.waitKey(1)
        if key == ord('q') :
            break

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.exit(main())

