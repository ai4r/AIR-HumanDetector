"""
Test openvino solution
"""

import cv2
import argparse
import time
import queue
from threading import Thread
import json
import logging as log
import os
import random
import sys



from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, DetectionsFromFileReader
from mc_tracker.mct import MultiCameraTracker
from utils.analyzer import save_embeddings
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from utils.video import MulticamCapture, NormalizerCLAHE
from utils.visualization import visualize_multicam_detections, get_target_size
from openvino_lib.openvino.inference_engine import IECore
#from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

from multi_camera_multi_target_tracking_demo import FramesThreadBody


set_log_config()


#SHOW_VIDEO=False
SHOW_VIDEO = True
WRITE_BBOX = False
#WRITE_BBOX = True


def save_all_video_res():
    # FOR certification
    parent_path = './data_origin/human_track_data_kaist_certification/video'
    out_parent_path = './res/exp1_cert'

    # FOR realenv
    #parent_path = './data_origin/human_track_data_kaist_realenv/video'
    #out_parent_path = './res/exp1'

    if not os.path.exists(out_parent_path):
        os.mkdir(out_parent_path)

    all_video = os.listdir(parent_path)

    for each_video in all_video:
        print('Processing on ' + each_video)
        each_video_abs_path = os.path.join(parent_path, each_video)
        out_video_abs_path = os.path.join(out_parent_path, each_video)

        if not os.path.exists(out_video_abs_path):
            os.mkdir(out_video_abs_path)
        demo_single_video(each_video_abs_path, out_video_abs_path)





def demo_single_video(video_path, output_path):
    #video_path = 'data/test_d-01-01-3p.mp4'
    #video_path = 'test_d-01-01-3p.mp4'
    #output_path = 'res/exp1/test_d-01-01-3p.mp4'

    detection_model_path = '/home/lhw/humancare/humanTrack/AIR-Human-Detector-master/openvino/model/person-detection-retail-0013/FP32/person-detection-retail-0013.xml'
    reidentification_model_path = 'model/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml'



    # init video capture
    capture = MulticamCapture([video_path], False)
    # load ie core
    ie = IECore()

    config = read_py_config('configs/person.py')

    #object_detector = DetectionsFromFileReader(detection_model_path, detector_th)
    object_detector = Detector(ie, detection_model_path, config['obj_det']['trg_classes'],
                               max_num_frames=capture.get_num_sources())
    object_recognizer = VectorCNN(ie, reidentification_model_path)

    run_single(config, capture, object_detector, object_recognizer,output_path)

def get_track_res_for_frame(all_objects):
    all_track_res = []
    for i, objects in enumerate(all_objects):
        for j, obj in enumerate(objects):
            left, top, right, bottom = obj.rect
            label = obj.label
            id = int(label.split(' ')[-1]) if isinstance(label, str)else int(label)
            if id >= 0:
                all_track_res.append((str(id), str(left), str(top), str(right), str(bottom)))

    return all_track_res




def write_bbox(curr_track_res, frame_idx, out_parent_path):
    obj_num = len(curr_track_res)
    if obj_num > 0:
        out_abs_path = os.path.join(out_parent_path, str(frame_idx) + '.csv')

        with open(out_abs_path, 'w') as wrtie_file:
            wrtie_file.write(str(obj_num) + '\n')
            for each_tracked in curr_track_res:
                id, left, top, right, bottom = each_tracked
                txt='body,{},init,{},{},{},{}\n'.format(id, left, top, right, bottom)
                wrtie_file.write(txt)


def run_single(config, capture, detector, reid, output_parent):
    win_name = 'Human detection and tracking'
    frame_number = 0
    avg_latency = AverageEstimator()
    key = -1

    if config['normalizer_config']['enabled']:
        capture.add_transform(
            NormalizerCLAHE(
                config['normalizer_config']['clip_limit'],
                config['normalizer_config']['tile_size'],
            )
        )

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config['sct_config'], **config['mct_config'],
                                 visual_analyze=config['analyzer'])

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)


    while thread_body.process:
        if SHOW_VIDEO:
            key = check_pressed_keys(key)
            if key == 27:
                break
        start = time.perf_counter()
        try:
            frames = thread_body.frames_queue.get_nowait()
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.wait_and_grab()

        frame_number += 1
        detector.run_async(frames, frame_number)

        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        latency = max(time.perf_counter() - start, sys.float_info.epsilon)
        avg_latency.update(latency)
        fps = round(1. / latency, 1)

        current_track_res = []

        vis = visualize_multicam_detections(prev_frames, tracked_objects, fps, **config['visualization_config'])


        current_track_res = get_track_res_for_frame(tracked_objects)
        if WRITE_BBOX:
            write_bbox(current_track_res, frame_number-1, output_parent)

        if SHOW_VIDEO:
            vis_resize = cv2.resize(vis, dsize=(0,0),fx=0.4, fy=0.4)
            cv2.imshow(win_name, vis_resize)


        print('\rProcessing frame: {}, fps = {} (avg_fps = {:.3})'.format(
            frame_number, fps, 1. / avg_latency.get()), end="")
        prev_frames, frames = frames, prev_frames
    print('')

    thread_body.process = False
    frames_thread.join()


    if len(config['embeddings']['save_path']):
        save_embeddings(tracker.scts, **config['embeddings'])

def main():
    #save_all_video_res()

    video_path = '../data/human_track_data_kaist_realenv/video/test_c-01-01-3p.mp4'
    #video_path = 'data_origin/human_track_data_kaist_realenv/video/test_d-02-05-3p.mp4'
    #video_path = 'data_origin/human_track_data_kaist_certification/video/test_a-02-02-2p.mp4'
    



    output_path = ''
    demo_single_video(video_path, output_path)

    """
    video_path = 'tmp/test_data/video/test_d-02-05-3p.mp4'
    output_path = ''
    demo_single_video(video_path, output_path)
    """

if __name__ == '__main__':
    main()






