"""
Demo for updated code
"""

import cv2
import os
import time
import sys
import cv2
import DataProvider
import numpy as np

from calcHistogram import calcHistogram3D
#from Demo_2_saveBox import TrackMod_dual, TrackMod, DET_THRESHOLD, MIN_DET_THRESHOLD

from Demo_2_saveBox import TrackMod, DET_THRESHOLD, MIN_DET_THRESHOLD

demo_mode = "VIDEO"
# demo_mode = "CAM"

VIDEO_PATH = './testSample/video/lab02-18-2p.mp4'
#VIDEO_PATH = './testVideo/test_a-01-06-2p.mp4'
#VIDEO_PATH = './testVideo/test_a-01-07-2p.mp4'



#VIDEO_PATH = './testVideo/test_a-01-09-2p.mp4'#OK
#VIDEO_PATH = './testVideo/test_b-01-01-2p.mp4'#One mistake
#VIDEO_PATH = './testVideo/test_b-02-02-2p.mp4'#Worst case
#VIDEO_PATH = './testVideo/test_b-03-02-2p.mp4'#OK
#VIDEO_PATH = './testVideo/test_a-05-10-2p.mp4'#Problem
VIDEO_PATH = './testVideo/test_a-02-03-2p.mp4'#BEST


def draw_panel(img_new, profile_list):
    """
        Draw panel including bbox result and profile with two tracker
    """

    # draw img
    panel_img = np.zeros(shape=(1000,1500, 3), dtype=np.uint8)


    img_new = cv2.resize(img_new, (640, 480))

    h1, w1, c1 = img_new.shape

    cv2.putText(panel_img, "New", (40, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)

    panel_img[20:20 + h1, 20:20 + w1, :] = img_new

    sub_img_size = (64, 64)

    for idx, each_profile in enumerate(profile_list):
        y_offset = 50 + img_new.shape[0] + idx * sub_img_size[1]
        x_init_offset = 20

        msg_id = str(each_profile.uid)

        # current saved image
        cv2.putText(panel_img, "Profile " + msg_id, (x_init_offset, y_offset+32), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
        curr_frame_max = 10
        for idx in range(curr_frame_max):
            len_profile = len(each_profile.rect_image_list)

            if idx >= len_profile:
                break

            if len_profile < curr_frame_max:
                curr_idx = idx
            else:
                curr_idx = len_profile - curr_frame_max + idx

            each_positive_img_rect = each_profile.rect_image_list[curr_idx]
            curr_img, curr_rect = each_positive_img_rect
            curr_cropped_img = curr_img[curr_rect[1]:curr_rect[1] + curr_rect[3],
                               curr_rect[0]:curr_rect[0] + curr_rect[2]]
            curr_cropped_img = cv2.resize(curr_cropped_img, sub_img_size)

            x_offset = x_init_offset + sub_img_size[0] * (idx + 1)
            panel_img[y_offset:y_offset + sub_img_size[1], x_offset:x_offset + sub_img_size[0], :] = curr_cropped_img

    return panel_img





def draw_panel(img, profile_list):
    """
        Draw panel including bbox result and profile
    """
    # draw img
    panel_img = np.zeros(shape=(900,  1500, 3), dtype=np.uint8)

    h1, w1, c1 = img.shape
    panel_img[20:20 + h1, 20:20 + w1, :] = img

    sub_img_size = (64, 64)

    for idx, each_profile in enumerate(profile_list):
        y_offset = 20 + img.shape[0] + idx * sub_img_size[1]
        x_init_offset = 20

        msg_id = str(each_profile.uid)

        # current saved image
        cv2.putText(panel_img, msg_id, (x_init_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        curr_frame_max = 10
        for idx in range(curr_frame_max):
            len_profile = len(each_profile.rect_image_list)

            if idx >= len_profile:
                break

            if len_profile < curr_frame_max:
                curr_idx = idx
            else:
                curr_idx = len_profile - curr_frame_max + idx

            each_positive_img_rect = each_profile.rect_image_list[curr_idx]
            curr_img, curr_rect = each_positive_img_rect
            curr_cropped_img = curr_img[curr_rect[1]:curr_rect[1] + curr_rect[3],
                               curr_rect[0]:curr_rect[0] + curr_rect[2]]
            curr_cropped_img = cv2.resize(curr_cropped_img, sub_img_size)

            x_offset = x_init_offset + sub_img_size[0] * (idx + 1)
            panel_img[y_offset:y_offset + sub_img_size[1], x_offset:x_offset + sub_img_size[0], :] = curr_cropped_img

    return panel_img

def demo_total(video_path):
    if demo_mode == "VIDEO":
        dp = DataProvider.VideoDataProvider(video_path)

    resize_rate = 0.4

    tracker = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                           min_det_treshold=MIN_DET_THRESHOLD,
                           use_cmatch=True, tracker_type='TM', tracker_limit=10)

    frame_idx = 0

    while True:
        img = dp.get()
        if img is None:
            break

        img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)
        img_resize_draw = img_resize.copy()

        tracker.run(img_resize)

        tracker.draw(img_resize_draw)

        panel = draw_panel(img_resize_draw,tracker.tracking.profile_classifier.profile_list)

        cv2.imshow("Tracking Result", panel)

        print(frame_idx)
        print('New')
        for trackRes in tracker.current_tracks:
            id = trackRes.id
            x1 = trackRes.tl[0]
            y1 = trackRes.tl[1]
            x2 = trackRes.br[0]
            y2 = trackRes.br[1]
            last_state = trackRes.last_state
            type = trackRes.type

            print(str(type) + ',' + str(id) + ',' + last_state + ',' + str(x1) + ',' + str(x2) + ',' + str(y1) + ',' + str(y2))



        frame_idx += 1

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key == ord('c'):
            cv2.waitKey()


def demo_single(video_path, old_tracker=False):
    if demo_mode == "VIDEO":
        dp = DataProvider.VideoDataProvider(video_path)

    resize_rate = 0.4
    if old_tracker:
        tracker = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                           min_det_treshold=MIN_DET_THRESHOLD,
                           use_cmatch=True, tracker_type='TM', tracker_limit=10, old_tracker=True)
    else:
        tracker = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                           min_det_treshold=MIN_DET_THRESHOLD,
                           use_cmatch=True, tracker_type='TM', tracker_limit=10, old_tracker=False)

    frame_idx = 0

    while True:
        img = dp.get()
        if img is None:
            break

        img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)
        img_resize_draw = img_resize.copy()

        tracker.run(img_resize)
        tracker.draw(img_resize_draw)

        """
        if frame_idx > 572:
            panel = draw_panel(img_resize_draw, tracker.tracking.profile_classifier.profile_list)
            cv2.imshow("Tracking Result", panel)
            cv2.waitKey()
        """

        if old_tracker:
            panel = draw_panel(img_resize_draw, [])
        else:
            panel = draw_panel(img_resize_draw, tracker.tracking.profile_classifier.profile_list)
        cv2.imshow("Tracking Result", panel)

        print(frame_idx)
        for trackRes in tracker.current_tracks:
            id = trackRes.id
            x1 = trackRes.tl[0]
            y1 = trackRes.tl[1]
            x2 = trackRes.br[0]
            y2 = trackRes.br[1]
            last_state = trackRes.last_state
            type = trackRes.type

            print(str(type) + ',' + str(id) + ',' + last_state + ',' + str(x1) + ',' + str(x2) + ',' + str(y1) + ',' + str(y2))

        frame_idx += 1

        key = cv2.waitKey(1)

        print(key)

        if key == ord('q'):
            break
        if key == ord('c'):
            cv2.waitKey()

if __name__ == "__main__":
    demo_total(VIDEO_PATH)
    #demo_single(VIDEO_PATH, old_tracker=True)