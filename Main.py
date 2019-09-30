import datetime
import cv2
import numpy as np
import pickle
import os, shutil, time

import xml.etree.ElementTree as XMLTree

from os import walk
from darkflow.net.build import TFNet
from darkflow.cli import argHandler

import tracking.Tracking as Tracking
import clustering.Grouping as Grouping

import DataProvider
import DataInput
import Utilities as util

from ELMCNN import ELMCNN
from NetworkDelegator import TensorflowNetwork
from yolo3.YOLOv3Delegator import YOLOv3Network

cfg = {}

def test_tracking():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    data_path_base = cfg['BasePath']
    min_conf_threshold = 0.3
    det_conf_threshold = 0.5
    IOU_threshold = 0.5

    dump_frame_length = 0
    dump_frame_jump = 900
    do_cleaning = True

    save_term = 0
    save_no = 0

    show_delay = 10
    feature_grid = [4, 4, -1]

    Tracking.TRACKER_TYPE = 'TM'
    Tracking.TRACKING_LIMIT = 5
    Tracking.KEEP_DATA_LENGTH = 5
    Tracking.USE_CMATCH = False

    use_yolo2 = False

    if not Tracking.USE_CMATCH:
        min_conf_threshold = det_conf_threshold


    if use_yolo2:
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": min_conf_threshold, 'gpu': 2.0}
        # options = {"model": "cfg/yolo-f.cfg", "load": 8000, "threshold": min_conf_threshold, 'gpu': 2.0}
        # options = {"metaLoad": "built_graph/yolo-body2.meta", "pbLoad": "built_graph/yolo-body2.pb", "threshold": 0.5, 'gpu': 2.0}
        # options = {"model": "cfg/yolo-body2.cfg", "load": 238500, "threshold": 0.5, 'gpu': 2.0}
        tfnet = TFNet(options)

    else:
        tfnet = YOLOv3Network('/home/youbit/storage/Workspaces/DarkflowYM/yolo3/person/model.ckpt-149',
                              '/home/youbit/storage/Workspaces/DarkflowYM/yolo3',
                              'person_classes.txt', 'person_anchors.txt',
                              min_conf_threshold)

    providers = [
        # dp = DataProvider.INRIADataProvider('../../Dataset/INRIAPerson')
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Chair'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Couch'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Follow'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Hallway'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting1'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting2'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting3'),
        # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting4')
        # DataProvider.ImageDataProvider('/home/youbit/Download/Google Images/sitting on sofa')
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 1),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 2),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 3),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 4),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 5),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 6),
        DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 7)
    ]

    results = []

    # Dump TP & FP result for clustering performance test
    TP_results = None
    FP_results = None
    # TP_results = []
    # FP_results = []


    shutil.rmtree('temp/tracking-error', ignore_errors=True)
    os.mkdir('temp/tracking-error')

    for dp_idx, dp in enumerate(providers):
        dump_list = []
        current_tracks = []
        frameSkip = 0

        P = 0
        DETECTION_TP = 0
        DETECTION_FP = 0
        CM_TP = 0
        CM_FP = 0
        TRACK_TP = 0
        TRACK_FP = 0
        MOTP_SUM = 0
        MOTA_MME = 0

        Tracking.lost_tracks = []
        Tracking.UID = 0

        while True:
            img = dp.get()
            if img is None:
                break

            if dp.frameNo < frameSkip:
                continue

            if dp.frameNo == 192:
                a = 1

            # dump_frame_length = dp.totalFrameNo

            disp_img = img.copy()

            if use_yolo2:
                result = tfnet.return_predict(img)
            else:
                result = tfnet.detect_image(img)

            '''
            for idx1, r1 in enumerate(result):
                for idx2, r2 in enumerate(result):
                    if idx1 == idx2:
                        continue
                    if r1['topleft']['x'] == r2['topleft']['x']  and r1['topleft']['y'] == r2['topleft']['y']\
                        and r1['bottomright']['x'] == r2['bottomright']['x'] and\
                        r1['bottomright']['y'] == r2['bottomright']['y']:
                        raise Exception('Something Wrong!! %s, %s ' % (str(r1), str(r2)))
            '''

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
                                   r['bottomright']['y'], conf, img)

                # High-Confidence Tracks
                if t.type == 'body':
                    if conf >= det_conf_threshold:
                        new_tracks.append(t)
                        new_tracks_rect.append(t.rect)
                    # Low-Confidence Tracks
                    elif conf >= min_conf_threshold:
                        candidate_tracks.append(t)
                        candidate_tracks_rect.append(t.rect)
                elif t.type == 'face':
                    if conf >= det_conf_threshold * 1:
                        new_tracks.append(t)
                        new_tracks_rect.append(t.rect)
                    # Low-Confidence Tracks
                    elif conf >= min_conf_threshold:
                        candidate_tracks.append(t)
                        candidate_tracks_rect.append(t.rect)

            # Setting for Candidate Match
            if Tracking.USE_CMATCH:
                new_tracks_features = tfnet.return_features(new_tracks_rect, feature_grid)
                for idx, tr in enumerate(new_tracks):
                    new_tracks[idx].feature = new_tracks_features[idx]

                candidate_tracks_features = tfnet.return_features(candidate_tracks_rect, feature_grid)
                for idx, tr in enumerate(candidate_tracks):
                    candidate_tracks[idx].feature = candidate_tracks_features[idx]

                Tracking.candidate_tracks = candidate_tracks
                Tracking.candidate_features = candidate_tracks_features

                # Setting for Tracking
                def get_feature(rect):
                    f = tfnet.return_features([rect], feature_grid)
                    return f

                Tracking.getFeature = get_feature

            # Update
            Tracking.update(current_tracks, new_tracks, img)

            # Data Cleaning?
            if dump_frame_length > 0:
                if dp.frameNo % dump_frame_length == 0:
                    print('Processing %d frames of %d total frames (%d%%)...' % (
                    dp.frameNo, dp.totalFrameNo, int((dp.frameNo * 100) / dp.totalFrameNo)))

                    # Dump data to file and stop
                    if not do_cleaning:
                        break
                    # Data Purification
                    elif len(dump_list) > 0:
                        do_data_cleaning(dump_list, type='body(f)', target_state='Search')
                        dump_list = []

                        if dump_frame_jump > 0 :
                            current_tracks = []
                            frameSkip = (dp.frameNo + dump_frame_jump)

                elif len(current_tracks) > 0:
                    if '' == '':
                        min_area_ratio = 0.05
                        max_area_ratio = 0.4
                        max_wh_ratio = 1.5
                        x_outer_padding = 0.1
                        y_outer_padding = 0.0
                    else:
                        min_area_ratio = 0.1
                        max_area_ratio = 0.25
                        max_wh_ratio = 1.0
                        x_outer_padding = 0.0
                        y_outer_padding = 0.0

                    tr_list = []
                    for t in current_tracks:
                        if False and t.last_state is 'Search':
                            continue

                        area_ratio = (t.rect[2] * t.rect[3]) / (img.shape[0] * img.shape[1])
                        wh_ratio = t.rect[2] / t.rect[3]

                        min_x = int(img.shape[1] * x_outer_padding)
                        max_x = int(img.shape[1] * (1 - x_outer_padding))
                        min_y = int(img.shape[0] * y_outer_padding)
                        max_y = int(img.shape[0] * (1 - y_outer_padding))

                        # Valid area ratio
                        if not (min_area_ratio < area_ratio < max_area_ratio):
                            continue

                        # Valid Width-Height ratio?
                        if wh_ratio > max_wh_ratio:
                            continue

                        # Valid area
                        if (t.rect[0] < min_x) or (t.rect[0] + t.rect[2] > max_x) or \
                                (t.rect[1] < min_y) or (t.rect[1] + t.rect[3] > max_y):
                            continue

                        t2 = t.copy()
                        t2.feature = t.feature
                        t2.frameNo = dp.frameNo
                        tr_list.append(t2)

                    if len(tr_list) > 0:
                        r = {'FrameNo': dp.frameNo, 'Image': img.copy(), 'Tracks': tr_list}
                        dump_list.append(r)

            # Detection / Tracking Performance
            if IOU_threshold > 0:
                # Count valid positive
                for t in dp.current_tracks:
                    if not t.is_outside:
                        P += 1

                # Check overlap-ratio and TP / FP
                for tr in current_tracks:
                    if not tr.type == 'body':
                        continue

                    is_TP = False
                    for pid, t in enumerate(dp.current_tracks):
                        if not t.is_marked and not (t.is_outside):
                            rect_1 = (tr.tl[0], tr.tl[1], tr.br[0], tr.br[1])
                            rect_2 = (t.tl[0], t.tl[1], t.br[0], t.br[1])
                            overlap = util.getOverlapRatio(rect_1, rect_2)

                            if overlap >= IOU_threshold:
                                is_TP = True
                                t.is_marked = True

                                # For MOTP
                                MOTP_SUM += (1 - overlap)

                                # For MOTA
                                if hasattr(tr, 'person_id') :
                                    if tr.person_id != pid:
                                        MOTA_MME += 1
                                tr.person_id = pid
                                break

                    if is_TP:
                        if tr.last_state == 'Match' or tr.last_state == 'Init':
                            DETECTION_TP += 1
                            tr.result = 'TP-DET'
                        elif tr.last_state == 'CMatch':
                            CM_TP += 1
                            tr.result = 'TP-CM'
                        elif tr.last_state == 'Search':
                            TRACK_TP += 1
                            tr.result = 'TP-TM'
                        else:
                            raise Exception('Programming Error!')

                    # False Positive
                    else:
                        if tr.last_state == 'Match' or tr.last_state == 'Init':
                            DETECTION_FP += 1
                            tr.result = 'FP-DET'
                        elif tr.last_state == 'CMatch':
                            CM_FP += 1
                            tr.result = 'FP-CM'
                        elif tr.last_state == 'Search':
                            TRACK_FP += 1
                            tr.result = 'FP-TM'
                        else:
                            raise Exception('Programming Error!')

                do_save = False
                for t in dp.current_tracks:
                    if not t.is_outside:
                        if t.is_marked:
                            cv2.rectangle(disp_img, t.tl, t.br, (255, 255, 255), 1)
                        else:
                            cv2.rectangle(disp_img, t.tl, t.br, (0, 0, 0), 1)
                            do_save = True

                for tr in current_tracks:
                    if tr.result[:2] == 'TP':
                        # For later benchmark
                        if TP_results is not None:
                            TP_results.append([util.crop(img, tr.rect), tr.confidence])

                        # For visualization
                        cv2.rectangle(disp_img, tr.tl, tr.br, (0, 255, 0), 1)
                    else:
                        # For later benchmark
                        if FP_results is not None:
                            FP_results.append([util.crop(img, tr.rect), tr.confidence])

                        # For visualization
                        cv2.rectangle(disp_img, tr.tl, tr.br, (0, 0, 255), 1)
                        cv2.putText(disp_img, tr.result, (tr.tl[0], tr.tl[1] - 15), cv2.FONT_HERSHEY_PLAIN, 0.8,
                                    (0, 0, 255), 1)
                        do_save = True

                if do_save:
                    cv2.imwrite('temp/tracking-error/%02d-%04d.jpg' % (dp_idx, dp.frameNo), disp_img)

            else:
                for tr in current_tracks:
                    tr.draw(disp_img)

            if save_term > 0 and dp.frameNo % save_term == 0 and len(current_tracks) > 0:
                save_no += 1
                save_trajectories('/media/data/PTMP/YL_Custom', img, current_tracks)
                print('Save %d...' % save_no)

            cv2.putText(disp_img, 'Frame : %d' % (dp.frameNo), (25, 25),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 255, 255))

            if show_delay >= 0:
                cv2.imshow('frame', disp_img)
                key = cv2.waitKey(show_delay)

                # ESC
                if key == 27:
                    break
                elif key == ord('1'):
                    show_delay = 0
                elif key == ord('2'):
                    show_delay = 100
                elif key == ord('3'):
                    show_delay = 20

                # Capture
                elif key == ord('s'):
                    save_trajectories('../../Dataset/YF', img, current_tracks)

        cv2.destroyAllWindows()

        if len(dump_list) > 0 and not do_cleaning:
            pickle.dump(dump_list, open('temp/result.pkl', 'wb'))
            print('%d results are dumped' % len(dump_list))

        if TP_results is not None:
            pickle.dump([TP_results, FP_results], open('temp/tracking_dump_%d.pickle' % dp_idx, 'wb'))

        if IOU_threshold > 0:
            TP = DETECTION_TP + CM_TP + TRACK_TP
            FP = DETECTION_FP + CM_FP + TRACK_FP
            PR = TP / (TP + FP)
            RC = TP / P

            result = [det_conf_threshold, IOU_threshold, PR, RC, P, TP, FP,
                        DETECTION_TP, DETECTION_FP, CM_TP, CM_FP, TRACK_TP, TRACK_FP,
                        MOTP_SUM, MOTA_MME]

            results.append(result)
            r = result[:]
            r[13] /= r[5]
            r[14] = 1 - (((r[4] - r[5]) + r[6] + r[14]) / r[4])
            print(r)

    if IOU_threshold > 0:
        all = [det_conf_threshold, IOU_threshold, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for r in results:
            for i in range(4, len(r)):
                all[i] += r[i]

                # MOTP
                if i == 13:
                    r[i] /= r[5]

                # MOTA
                if i == 14:
                    r[i] = 1 - (((r[4] - r[5]) + r[6] + r[i]) / r[4])

        ALL_TP = all[5]
        ALL_FP = all[6]
        ALL_P = all[4]

        # Total Precision
        all[2] = ALL_TP / (ALL_TP + ALL_FP)

        # Total Recall
        all[3] = ALL_TP / ALL_P

       # Total MOTP
        all[13] /= ALL_TP

        # MOTA
        ALL_MISS = ALL_P - ALL_TP
        ALL_MME = all[14]
        all[14] = 1 - ((ALL_MISS + ALL_FP + ALL_MME) / ALL_P)

        results.append(all)
        print('------------------------------------------')
        print(all)

        util.save_csv(results, 'temp/tracking_result.csv')

def record_tracking_result(save_dir, resize_rate):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    min_conf_threshold = 0.3
    det_conf_threshold = 0.5
    do_save_background = True

    feature_grid = [4, 4, -1]

    Tracking.BOUNDINGBOX_WEIGHT = 0.2
    Tracking.CENTER_OBJECT_CONFIDENCE_THRESHOLD = 0.0
    Tracking.KEEP_DATA_LENGTH = 5
    Tracking.TRACKING_LIMIT = 60
    Tracking.TRACKER_TYPE = 'TM'
    Tracking.USE_CMATCH = True

    use_yolo2 =True

    if not Tracking.USE_CMATCH:
        min_conf_threshold = det_conf_threshold

    if use_yolo2:
        # options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": min_conf_threshold, 'gpu': 2.0}
        options = {"model": "cfg/yolo-f.cfg", "load": 8000, "threshold": min_conf_threshold, 'gpu': 2.0}
        # options = {"model": "cfg/yolo-coco-f4.cfg", "load": 31000, "threshold": min_conf_threshold, 'gpu': 2.0}
        # options = {"metaLoad": "built_graph/yolo-body2.meta", "pbLoad": "built_graph/yolo-body2.pb", "threshold": 0.5, 'gpu': 2.0}
        # options = {"model": "cfg/yolo-body2.cfg", "load": 238500, "threshold": 0.5, 'gpu': 2.0}
        # options = {"model": "cfg/VOC.cfg", "load": 434, "threshold": 0.5, 'gpu': 2.0}
        tfnet = TFNet(options)

    else:
        tfnet = YOLOv3Network('/home/youbit/storage/Workspaces/DarkflowYM/yolo3/person/model.ckpt-149',
                              '/home/youbit/storage/Workspaces/DarkflowYM/yolo3',
                              'person_classes.txt', 'person_anchors.txt',
                              min_conf_threshold)

    target_db = 2
    bg_path = '/media/data/ETRI-HumanCare/LivingLabBG'

    if do_save_background:
        shutil.rmtree(bg_path, ignore_errors=True)
        os.mkdir(bg_path)

    if target_db == 1:
        providers = [
            # dp = DataProvider.INRIADataProvider('../../Dataset/INRIAPerson')
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Chair'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Couch'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Follow'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Hallway'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting1'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting2'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting3'),
            # DataProvider.IllmenauDataProvider(data_path_base + '/PeopleTracking', 'Sitting4')
            # DataProvider.ImageDataProvider('/home/youbit/Download/Google Images/sitting on sofa')
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 1),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 2),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 3),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 4),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 5),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 6),
            DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 7)
        ]

    # Scan multiple files from directories
    if target_db == 2:
        providers = []
        parent_dir_list = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호'
        ]
        dir_list = []
        for dir in parent_dir_list:
            for (dirpath, dirnames, filenames) in walk(dir):
                dir_list.append(dirpath)

        for dir in dir_list:
            for (dirpath, dirnames, filenames) in walk(dir):
                for f_name in filenames:
                    if f_name.endswith('.avi'):
                        dp = DataProvider.VideoDataProvider('%s/%s' % (dirpath, f_name), resize_rate=0.5)
                        providers.append(dp)


    # Add Single file
    if '' == 'D':
        file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180423AM_S02_H120_P07_A26-115709.avi']

    if target_db == 3:
        providers = []
        for i in range(7):
            test_case = i + 1
            if test_case == 1:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V08-170634.avi']
                start_index = [0]
            elif test_case == 2:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180423AM_S02_H120_P07/20180423AM_S02_H120_P07_A26-115709.avi']
                start_index = [0]
            elif test_case == 3:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180420PM_S02_H120_P07/20180420PM_S02_H120_P07_A24-145932.avi']
                start_index = [0]
            elif test_case == 4:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V01-133626.avi']
                start_index = [34644]
            elif test_case == 5:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V08-170634.avi']
                start_index = [5696]
            elif test_case == 6:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180420PM_S02_H120_P07/20180420PM_S02_H120_P07_A18-150830.avi']
                start_index = [0]
            elif test_case == 7:
                file_names = ['/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180425PM_S02_H070_P07/20180425PM_S02_H070_P07_A47-155715.avi']
                start_index = [50]

            st_idx = start_index[0]
            en_dix = st_idx + (24 * 60 * 20)
            dp = DataProvider.VideoDataProvider(file_names[0], resize_rate=0.5, start_frame=st_idx, end_frame=en_dix)
            providers.append(dp)


    for dp_idx, dp in enumerate(providers):
        current_tracks = []
        Tracking.UID = 0
        vwt = None
        print('[%d/%d] ...' % (dp_idx+1, len(providers)))

        while True:
            img = dp.get()

            if img is None:
                break

            if resize_rate != 1.0:
                img = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            # img = cv2.transpose(img)
            # img = cv2.flip(img, 0, 1)

            if use_yolo2:
                result = tfnet.return_predict(img)
            else:
                result = tfnet.detect_image(img)

            new_tracks = []
            new_tracks_rect = []
            candidate_tracks = []
            candidate_tracks_rect = []

            # Save background images for background image learning
            if do_save_background:
                if len(result) == 0:
                    img_file = '%s/%d-%d.jpg' % (bg_path, dp_idx, dp.frameNo)

                    '''
                    cv2.imshow('Background', img)
                    cv2.waitKey(1)
                    '''

                    cv2.imwrite(img_file, img)
                    print(img_file)
                else:
                    '''
                    for r in result:
                        cv2.rectangle(img, (r['topleft']['x'], r['topleft']['y']),
                                      (r['bottomright']['x'], r['bottomright']['y']), (255, 255, 255), 1)

                    cv2.imshow('Foreground', img)
                    cv2.waitKey(1)c
                    '''

                if dp.frameNo % 100 == 0:
                    print('%d frame ...' % dp.frameNo)

                continue

            for r in result:
                class_name = r['label']
                conf = r['confidence']

                if not class_name in ['person', 'body', 'face', 'body(u)', 'body(b)']:
                    continue

                if class_name == 'person':
                    class_name = 'body'

                t = Tracking.Track(class_name, r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'],
                                   r['bottomright']['y'], conf, img)

                # High-Confidence Tracks
                if t.type == 'body' or True:
                    if conf >= det_conf_threshold:
                        new_tracks.append(t)
                        new_tracks_rect.append(t.rect)
                    # Low-Confidence Tracks
                    elif conf >= min_conf_threshold:
                        candidate_tracks.append(t)
                        candidate_tracks_rect.append(t.rect)
                '''
                elif t.type == 'face':
                    if conf >= det_conf_threshold * 1:
                        new_tracks.append(t)
                        new_tracks_rect.append(t.rect)
                    # Low-Confidence Tracks
                    elif conf >= min_conf_threshold:
                        candidate_tracks.append(t)
                        candidate_tracks_rect.append(t.rect)
                '''

            # Setting for Candidate Match
            new_tracks_features = tfnet.return_features(new_tracks_rect, feature_grid)
            for idx, tr in enumerate(new_tracks):
                new_tracks[idx].feature = new_tracks_features[idx]

            candidate_tracks_features = tfnet.return_features(candidate_tracks_rect, feature_grid)
            for idx, tr in enumerate(candidate_tracks):
                candidate_tracks[idx].feature = candidate_tracks_features[idx]

            Tracking.candidate_tracks = candidate_tracks
            Tracking.candidate_features = candidate_tracks_features

            # Setting for Tracking
            def get_feature(rect):
                f = tfnet.return_features([rect], feature_grid)
                return f

            Tracking.getFeature = get_feature

            # Update
            Tracking.update(current_tracks, new_tracks, img)

            cv2.putText(img, 'Frame : %d' % dp.frameNo, (50, 55), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))
            cv2.putText(img, 'Frame : %d' % dp.frameNo, (55, 45), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0))

            def draw_rect(tr):
                if tr.last_state in ['Match', 'Init']:
                    color = (0, 255, 0)
                elif tr.last_state == 'CMatch':
                    color = (0, 255, 255)
                elif tr.last_state == 'Search':
                    color = (255, 0, 0)
                else:
                    raise Exception('Invalid state = ' + tr.last_state)

                thick = 3
                if tr.type == 'body':
                    cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0], tr.tl[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                (0, 0, 0))
                    cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0], tr.tl[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.0,
                                (255,255,255))
                else:
                    thick = 1
                    color = (255, 255, 255)

                cv2.rectangle(img, tr.tl, tr.br, color, thick)


            for tr in current_tracks:
                draw_rect(tr)
                if tr.face is not None:
                    draw_rect(tr.face)
                if tr.upper is not None:
                    draw_rect(tr.upper)
                if tr.bottom is not None:
                    draw_rect(tr.bottom)

            if 'D' == 'D':
                cv2.imshow('Result', img)
                cv2.waitKey(1)

            if vwt is None:
                vwt = cv2.VideoWriter('%s/%d.avi' % (save_dir, dp_idx), cv2.VideoWriter_fourcc(*'XVID'), float(24),
                                      (img.shape[1], img.shape[0]), True)

            vwt.write(img)

        if vwt is not None:
            vwt.release()


def test_detection(test_db='Illmenau', filename='', tfnet=None):
    target_conf = [0.1]
    target_IOU = [0.5]

    show_delay = -1

    if tfnet is None:
        # options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.0, 'gpu': 2.0}
        # options = {"model": "cfg/yolo-voc.cfg", "load": "bin/yolo-voc.weights", "threshold": 0.0, 'gpu': 2.0}
        options = {"model": "cfg/yolo-f.cfg", "load": 8000, "threshold": 0.0, 'gpu': 1.0}
        # options = {"model": "cfg/yolo-f-old.cfg", "load": 8000, "threshold": 0.0, 'gpu': 1.0}
        # options = {"model": "cfg/yolo-voc-f0.cfg", "load": 2000, "threshold": 0.0, 'gpu': 1.0}
        # options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.0, 'gpu': 2.0}
        tfnet = TFNet(options)

    if test_db == 'Illmenau':
        target_names = ['Chair', 'Couch', 'Follow', 'Hallway', 'Sitting1', 'Sitting2', 'Sitting3', 'Sitting4']

    elif test_db == 'VOC':
        target_names = ['VOC-Train', 'VOC-Test']

    elif test_db == 'VOC-R':
        target_names = ['VOC-Train-R', 'VOC-Test-R']

    else:
        raise Exception('Invalid test_db = ' + str(test_db))

    results = []
    for t_IOU in target_IOU:
        for t_conf in target_conf:
            P = 0
            TP = 0
            FP = 0
            for t_name in target_names:
                ret = [t_name, t_conf, t_IOU]
                print('Test ' + str(ret) + ' ...')

                tfnet.FLAGS.threshold = t_conf
                det = do_test_detection(t_name, t_conf, t_IOU, tfnet, show_delay)
                P += det[2]
                TP += det[3]
                FP += det[4]
                print('Result : ' + str(det))

                ret.extend(det)
                results.append(ret)

            if test_db == 'Illmenau' and (TP + FP) > 0:
                all_result = ['All', t_conf, t_IOU, TP / (TP + FP), TP / P, P, TP, FP]
                print('All : ' + str(all_result))
                results.append(all_result)

    if filename == '':
        filename = 'temp/detection_result.csv'
        util.save_csv(results, filename)
    return results


def do_test_detection(target, conf_threshold, overlap_threshold, tfnet, show_delay=0):
    data_path_base = cfg['BasePath']
    if target == 'VOC-Train':
        dp = DataProvider.VOCDataProvider(data_path_base + 'VOC2012/TrainVal/All')

    elif target == 'VOC-Test':
        dp = DataProvider.VOCDataProvider(data_path_base + 'VOC2012/Test')

    elif target == 'VOC-Train-R':
        dp = DataProvider.VOCDataProvider(data_path_base + 'VOC2012R/Train')

    elif target == 'VOC-Test-R':
        dp = DataProvider.VOCDataProvider(data_path_base + 'VOC2012R/Test')

    else:
        dp = DataProvider.IllmenauDataProvider(data_path_base + 'PeopleTracking', target)

    P = 0
    TP = 0
    FP = 0
    while True:
        img = dp.get()
        if img is None:
            break

        result = tfnet.return_predict(img)

        for t in dp.current_tracks:
            if not t.is_outside:
                P += 1

        for r in result:
            name = r['label']

            # We only consider 'person'
            if not name in ['person', 'body']:
                continue

            if name == 'body':
                name = 'person'

            conf = r['confidence']
            rect_1 = (r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'], r['bottomright']['y'])

            is_TP = False
            if conf >= conf_threshold:
                for t in dp.current_tracks:
                    if name == t.name and not t.is_marked and not (t.is_outside):
                        rect_2 = (t.tl[0], t.tl[1], t.br[0], t.br[1])
                        overlap = util.getOverlapRatio(rect_1, rect_2)

                        if overlap >= overlap_threshold:
                            is_TP = True
                            t.is_marked = True
                            break

            if is_TP:
                TP += 1
            else:
                FP += 1

            if show_delay >= 0:
                if is_TP:
                    cv2.rectangle(img, (rect_1[0], rect_1[1]), (rect_1[2], rect_1[3]), (0, 255, 0), 1)
                else:
                    cv2.rectangle(img, (rect_1[0], rect_1[1]), (rect_1[2], rect_1[3]), (0, 0, 255), 1)

                for t in dp.current_tracks:
                    if not t.is_outside:
                        cv2.rectangle(img, t.tl, t.br, (255, 255, 255), 1)

        if show_delay >= 0:
            cv2.imshow('Result', img)
            cv2.waitKey(show_delay)

    if TP + FP > 0 and P > 0:
        PR = TP / (TP + FP)
        RC = TP / P
    else:
        PR = 0.0
        RC = 0.0
    return [PR, RC, P, TP, FP]

def load_config(filename):
    global cfg
    key = ''
    value = ''
    for line in open(filename):
        if line.strip() == '':
            if key != '' and value != '':
                cfg[key] = value
                key = ''
                value = ''
            continue
        elif line[0] == '[':
            key = line.strip()[1:-1]
        else:
            value += line.strip()

    if key != '' and value != '':
        cfg[key] = value

def save_trajectory(dir, img, b_rect=None, f_rect=None):
    filename = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cv2.imwrite('%s/Images/%s.jpg' % (dir, filename), img)
    xmlRoot = XMLTree.Element('annotation')

    n = XMLTree.Element('folder')
    n.text = 'YF'
    xmlRoot.append(n)

    n = XMLTree.Element('filename')
    n.text = filename + '.jpg'
    xmlRoot.append(n)

    n = XMLTree.Element('size')
    XMLTree.SubElement(n, 'width').text = str(img.shape[1])
    XMLTree.SubElement(n, 'height').text = str(img.shape[0])
    XMLTree.SubElement(n, 'depth').text = str(img.shape[2])
    xmlRoot.append(n)

    if b_rect is not None:
        n = XMLTree.Element('object')
        XMLTree.SubElement(n, 'name').text = 'person'

        bndbox = XMLTree.Element('bndbox')
        XMLTree.SubElement(bndbox, 'xmin').text = str(b_rect[0])
        XMLTree.SubElement(bndbox, 'ymin').text = str(b_rect[1])
        XMLTree.SubElement(bndbox, 'xmax').text = str(b_rect[0] + b_rect[2])
        XMLTree.SubElement(bndbox, 'ymax').text = str(b_rect[1] + b_rect[3])
        n.append(bndbox)
        xmlRoot.append(n)

    if f_rect is not None:
        n = XMLTree.Element('object')
        XMLTree.SubElement(n, 'name').text = 'face'

        bndbox = XMLTree.Element('bndbox')
        XMLTree.SubElement(bndbox, 'xmin').text = str(f_rect[0])
        XMLTree.SubElement(bndbox, 'ymin').text = str(f_rect[1])
        XMLTree.SubElement(bndbox, 'xmax').text = str(f_rect[0] + f_rect[2])
        XMLTree.SubElement(bndbox, 'ymax').text = str(f_rect[1] + f_rect[3])
        n.append(bndbox)
        xmlRoot.append(n)

    util.indentXML(xmlRoot)
    XMLTree.ElementTree(xmlRoot).write('%s/Annotations/%s.xml' % (dir, filename))


def save_trajectories(dir, img, trajectories, file_no=-1):
    if len(trajectories) == 0:
        return

    if file_no < 0:
        import time
        millis = str(round(time.time() * 1000))
        filename = millis
    else:
        filename = '%08d' % file_no
    cv2.imwrite('%s/Images/%s.jpg' % (dir, filename), img)
    xmlRoot = XMLTree.Element('annotation')

    n = XMLTree.Element('folder')
    n.text = 'YF'
    xmlRoot.append(n)

    n = XMLTree.Element('filename')
    n.text = filename + '.jpg'
    xmlRoot.append(n)

    n = XMLTree.Element('size')
    XMLTree.SubElement(n, 'width').text = str(img.shape[1])
    XMLTree.SubElement(n, 'height').text = str(img.shape[0])
    XMLTree.SubElement(n, 'depth').text = str(img.shape[2])
    xmlRoot.append(n)

    for tr in trajectories:
        n = XMLTree.Element('object')
        XMLTree.SubElement(n, 'name').text = tr.type

        bndbox = XMLTree.Element('bndbox')
        XMLTree.SubElement(bndbox, 'xmin').text = str(tr.tl[0])
        XMLTree.SubElement(bndbox, 'ymin').text = str(tr.tl[1])
        XMLTree.SubElement(bndbox, 'xmax').text = str(tr.br[0])
        XMLTree.SubElement(bndbox, 'ymax').text = str(tr.br[1])
        n.append(bndbox)
        xmlRoot.append(n)

    util.indentXML(xmlRoot)
    XMLTree.ElementTree(xmlRoot).write('%s/Annotations/%s.xml' % (dir, filename))


def save_annotations(dir, img, body_list, face_list, filename=''):
    if filename == '':
        import time
        millis = str(round(time.time() * 1000))
        filename = millis

    file_ext = '.jpg'

    cv2.imwrite('%s/Images/%s%s' % (dir, filename, file_ext), img)
    xmlRoot = XMLTree.Element('annotation')

    n = XMLTree.Element('folder')
    n.text = 'Custom'
    xmlRoot.append(n)

    n = XMLTree.Element('filename')
    n.text = filename + file_ext
    xmlRoot.append(n)

    n = XMLTree.Element('size')
    XMLTree.SubElement(n, 'width').text = str(img.shape[1])
    XMLTree.SubElement(n, 'height').text = str(img.shape[0])
    XMLTree.SubElement(n, 'depth').text = str(img.shape[2])
    xmlRoot.append(n)

    for b in body_list:
        n = XMLTree.Element('object')
        XMLTree.SubElement(n, 'name').text = 'person'

        bndbox = XMLTree.Element('bndbox')
        XMLTree.SubElement(bndbox, 'xmin').text = str(b[0])
        XMLTree.SubElement(bndbox, 'ymin').text = str(b[1])
        XMLTree.SubElement(bndbox, 'xmax').text = str(b[2])
        XMLTree.SubElement(bndbox, 'ymax').text = str(b[3])
        n.append(bndbox)
        xmlRoot.append(n)

    for f in face_list:
        n = XMLTree.Element('object')
        XMLTree.SubElement(n, 'name').text = 'face'

        bndbox = XMLTree.Element('bndbox')
        XMLTree.SubElement(bndbox, 'xmin').text = str(f[0])
        XMLTree.SubElement(bndbox, 'ymin').text = str(f[1])
        XMLTree.SubElement(bndbox, 'xmax').text = str(f[2])
        XMLTree.SubElement(bndbox, 'ymax').text = str(f[3])
        n.append(bndbox)
        xmlRoot.append(n)

    util.indentXML(xmlRoot)
    XMLTree.ElementTree(xmlRoot).write('%s/Annotations/%s.xml' % (dir, filename))


def use_opencv_tracker():
    output_dir = '/media/data/ETRI-HumanCare/LivingLabSet'
    start_index = [6616]
    capture_length = [24 * 120]

    test_case = 8
    if test_case == 1:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V08-170634.avi']
        start_index = [0]
    elif test_case == 2:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180423AM_S02_H120_P07/20180423AM_S02_H120_P07_A26-115709.avi']
        start_index = [0]
    elif test_case == 3:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180420PM_S02_H120_P07/20180420PM_S02_H120_P07_A24-145932.avi']
        start_index = [0]
    elif test_case == 4:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V01-133626.avi']
        start_index = [34644]
    elif test_case == 5:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180416PM_S01_H120_P07/20180416PM_S01_H120_P07_V08-170634.avi']
        start_index = [5696]
    elif test_case == 6:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180420PM_S02_H120_P07/20180420PM_S02_H120_P07_A18-150830.avi']
        start_index = [0]
    elif test_case == 7:
        file_names = [
            '/media/data/ETRI-HumanCare/LivingLab/2018년도 7차 김차순 할머니 충선로 8번길 52 예성A 305호/20180425PM_S02_H070_P07/20180425PM_S02_H070_P07_A47-155715.avi']
        start_index = [50]

    shutil.rmtree('%s/%d/Annotations' % (output_dir, test_case), ignore_errors=True)
    shutil.rmtree('%s/%d/Images' % (output_dir, test_case), ignore_errors=True)
    os.mkdir('%s/%d/Annotations' % (output_dir, test_case))
    os.mkdir('%s/%d/Images' % (output_dir, test_case))

    start_frame = start_index[0]
    end_frame = start_frame + capture_length[0]

    dp = DataProvider.VideoDataProvider(file_names[0])
    dp.jump(start_frame)

    class Param():
        pass

    param = Param()
    param.roi = None
    param.do_init = False
    param.state = 'Normal'
    param.cursor = [0, 0]

    regions = []
    region_idx = 0

    trackers = []

    def select_roi(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if param.state == 'Normal':
                param.state = 'Select'
                param.roi = [x, y, 0, 0]

        elif event == cv2.EVENT_MOUSEMOVE:
            if param.state == 'Select':
                param.roi[2] = x - param.roi[0]
                param.roi[3] = y - param.roi[1]
            param.cursor = [x, y]

        if event == cv2.EVENT_LBUTTONUP:
            if param.state == 'Select':
                if param.roi[2] > 0 and param.roi[3] > 0:
                    if len(regions) > region_idx:
                        regions[region_idx] = param.roi
                        t = Tracking.KCFTracker.Tracker(resized_img,
                                                        (param.roi[0], param.roi[1], param.roi[2], param.roi[3]))
                        trackers[region_idx] = t

                param.state = 'Normal'


    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', select_roi, param)

    is_running = True
    while is_running:
        original_img = dp.get()

        if original_img is None or dp.frameNo > end_frame:
            break

        resized_img = cv2.resize(original_img, (0,0), fx=0.5, fy=0.5)
        h, w, _ = resized_img.shape

        def draw_info(img):
            cv2.putText(img, 'Frame: %d' % dp.frameNo, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255))

            cv2.line(img, (param.cursor[0], 0), (param.cursor[0], h), (255, 255, 255), 1)
            cv2.line(img, (0, param.cursor[1]), (w, param.cursor[1]), (255, 255, 255), 1)

        def draw_select(img):
            if param.state == "Select":
                cv2.rectangle(img, (param.roi[0], param.roi[1]),
                              (param.roi[0] + param.roi[2], param.roi[1]+param.roi[3]), (0, 0, 255), 1)


        def draw_regions(img):
            colors = [(0, 255, 0), (255, 0, 0)]
            for idx, r in enumerate(regions):
                if r[0] < 0:
                    continue
                cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), colors[idx], 1)

        # Tracking update
        for idx, t in enumerate(trackers):
            new_roi = t.update(resized_img)
            if new_roi is not None:
                regions[idx] = [new_roi[0], new_roi[1], new_roi[2], new_roi[3]]
            else:
                regions[idx][0] = -1

        while True:
            disp_img = resized_img.copy()

            draw_info(disp_img)
            draw_select(disp_img)
            draw_regions(disp_img)

            cv2.imshow('Image', disp_img)
            key = cv2.waitKey(1)

            # ESC - Exit
            if key == 27:
                is_running = False
                break
            # Space - Task complete
            elif key == 32:
                new_regions = []
                for r in regions:
                    r = [r[0] * 2, r[1] * 2, r[2] * 2, r[3] * 2]
                    r = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
                    new_regions.append(r)
                save_annotations('%s/%d' % (output_dir, test_case), original_img, new_regions, [], '%05d' % dp.frameNo)
                break

            # Set ROI and Tracker for Person 1
            elif key == ord('1'):
                if len(regions) > 0:
                    regions[0][0] = -1
                else:
                    regions.append([-1, 0, 0, 0])
                    trackers.append(None)
                region_idx = 0

            # Set ROI and Tracker for Person 2
            elif key == ord('2'):
                if len(regions) > 1:
                    region_idx = 1
                    regions[1][0] = -1
                else:
                    while len(regions) < 2:
                        regions.append([-1, 0, 0, 0])
                        trackers.append(None)
                region_idx = 1

            # Reset
            elif key == ord('3'):
                regions = []
                trackers = []


def train_and_test():
    data_path_base = cfg['BasePath']

    model = 'cfg/voc(r)-f1-a1.cfg'
    load_count = 0
    save_step = 1000
    test_step = 1000
    gpu_usage = 1.0
    gpu_sleep = 0

    # path_name = 'Custom_Sitting'
    # path_name = 'Sitting+Walking'
    path_name = 'Sitting+VOC+INRIA+Walking+YLF'
    # path_name = 'Sitting+VOC+Walking'
    # path_name = 'SELF/Exercise'
    # path_name = 'VOC2012R/Train'
    # path_name = 'Sitting+VOC'
    # path_name = 'VOC2012/TrainVal/All'

    if load_count > 0:
        load_count = str(load_count)
    elif load_count == 0:
        load_count = 'bin/yolo-voc.weights'

    train_args = [
        'flow',
        '--train',
        '--model', model,
        '--dataset', data_path_base + path_name + '/Images',
        '--annotation', data_path_base + path_name + '/Annotations',
        '--momentum', '0.0',
        '--save', str(save_step),
        '--test_step', str(test_step),
        '--gpu', str(gpu_usage),
        '--gpuName', '/gpu:0',
        '--keep', '0'
    ]

    if load_count != -1 :
        train_args.append('--load')
        train_args.append(load_count)

    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(train_args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)

    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup,
              os.path.join(FLAGS.imgdir, 'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try:
        FLAGS.load = int(FLAGS.load)
    except:
        pass

    tfnet = TFNet(FLAGS)

    tfnet.best_record = None
    util.record_time()

    def test_func(step):
        is_best = False
        results = test_detection('VOC-R', '', tfnet)
        for i in range(len(results)) :
            results[i].insert(0, step)

        results[-1].append(util.get_elapsed_time())

        current_record = results[-1]
        best_f1 = 0
        if tfnet.best_record is not None:
            s = (tfnet.best_record[4] + tfnet.best_record[5])
            m = (tfnet.best_record[4] * tfnet.best_record[5])
            if s > 0:
                best_f1 = (2 * m) / s
            else:
                best_f1 = 0

        s = (current_record[4] + current_record[5])
        m = (current_record[4] * current_record[5])

        if s > 0:
            current_f1 = (2 * m) / s
        else:
            current_f1 = 0

        print("Current: %.3f / Best: %.3f" % (current_f1, best_f1))
        if current_f1 > best_f1:
            tfnet.best_record = current_record
            is_best = True
            util.save_csv([tfnet.best_record], 'temp/(%s) best_record.csv' % (model[4:-4]), is_append=True)

        util.save_csv(results, 'temp/(%s) detection.csv' % (model[4:-4]), is_append=True)
        if gpu_sleep > 0:
            print('Sleep %d seconds for cooling down' % gpu_sleep)
            time.sleep(gpu_sleep)

        util.record_time()
        return is_best

    tfnet.train(test_func=test_func)


def do_clustering(data_list, target_type='body', target_state='Match',
                  img_width=32, img_height=32, rank_k=10, temporal_k=0,
                  cluster_ratio=0.2, min_avg_conf=0.8, min_cluster_size=3,
                  use_HOG=False, use_DeepFeature=False):
    images = []
    features = []
    confs = []
    frameNo = []
    seqNo = []
    trackNo = []

    fd = cv2.HOGDescriptor()

    for idx, dp in enumerate(data_list):
        img = dp['Image']
        tracks = dp['Tracks']

        for track_idx, t in enumerate(tracks):
            rect = None
            if target_type == 'body' and t.type == 'body':
                if target_state == 'Match' and t.last_state is not 'Search':
                    rect = t.rect
                elif target_state == 'Search':
                    rect = t.rect

            elif target_type == 'face' and t.type == 'body':
                if t.face is not None:
                    f = t.face
                    if target_state == 'Match' and f.last_state is not 'Search':
                        rect = f.rect
                    elif target_state == 'Search':
                        rect = f.rect

            if rect is None:
                continue

            sub_img = util.crop(img, rect)

            # sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            sub_img = cv2.resize(sub_img, (img_width, img_height))
            # sub_img = cv2.resize(sub_img, (32, 32))

            # Raw images
            # cv2.normalize(sub_img, sub_img, 0, 1, cv2.NORM_MINMAX)
            images.append(sub_img)
            confs.append(t.confidence)

            frameNo.append(t.frameNo)
            seqNo.append(idx)
            trackNo.append(track_idx)

            # Deep features from Detector
            if use_DeepFeature:
                features.append(t.feature)

            # HOG
            if use_HOG:
                f = fd.compute(sub_img)
                features.append(f)

    if len(images) == 0:
        return None, None, None

    images = np.array(images)
    conf = np.array(confs)

    cluster_k = int(len(images) * cluster_ratio)
    if cluster_k < 1:
        cluster_k = 1

    if len(features) > 0:
        features = np.array(features)

    ds = DataInput.DataSets()

    if len(features) == 0:
        ds.train = DataInput.DataSet(images, None, is_image=True)
    else:
        ds.train = DataInput.DataSet(features, None, is_image=False)

    cluster_0 = Grouping.do_ROClustering(rank_k, cluster_k, temporal_k, conf, frameNo, ds)

    cluster_1 = []
    for cluster_idx, cluster in enumerate(cluster_0):
        avg_conf = cluster.average_confidence
        if avg_conf > min_avg_conf and cluster.num_examples >= min_cluster_size:
            cluster_1.append(cluster)

    for c in cluster_1:
        c.frame_index = []
        c.seq_index = []
        c.track_index = []

        for idx in c.index:
            c.frame_index.append(frameNo[idx])
            c.seq_index.append(seqNo[idx])
            c.track_index.append(trackNo[idx])

    return cluster_1, images, conf


def test_outlier_removal(dir_path):
    images = []
    confs = []

    for filePath in os.listdir(dir_path):
        img = cv2.imread(dir_path + '/' + filePath)
        img = np.float32(img)
        cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)
        images.append(img)

        length = len(filePath)
        conf = filePath[length - 9:length - 5]
        confs.append(float(conf))

    ret, err = do_outlier_removal(images, 0.1, 10, 100)

    for i in range(len(ret)):
        cv2.imwrite('temp/cluster/%d-%d.jpg' % (ret[i], i), np.uint8(images[i] * 255))

def do_outlier_removal(images, node_ratio, min_iter, max_iter):
    idx, err = Grouping.do_OutlierRemove(node_ratio, min_iter, max_iter, images)
    return idx, err

def test_cluster():
    if os.path.exists('temp/cluster'):
        shutil.rmtree('temp/cluster')

    print('Loading data ...')
    dump_list = pickle.load(open('temp/result.pkl', 'rb'))

    clusters, images, confs = do_clustering(dump_list, target_type='body',
                                            target_state='Match',
                                            min_avg_conf=0.0,
                                            cluster_ratio=0.1,
                                            img_width=64, img_height=192)

    for cluster_idx, cluster in enumerate(clusters):
        dir_path = 'temp/cluster/%d (%.2f)' % (cluster_idx, cluster.average_confidence)
        os.makedirs(dir_path)

        for item_idx, item in enumerate(cluster.index):
            conf = confs[item]
            sub_image = images[item]

            seq_idx = cluster.seq_index[item_idx]
            track_idx = cluster.track_index[item_idx]
            # conf = cluster.confidence[item_idx]
            track = dump_list[seq_idx]['Tracks'][track_idx]
            # image = dump_list[seq_idx]['Image']
            # sub_image = util.crop(image, track.rect)

            if track.last_state == 'Match':
                cv2.imwrite('%s/%d_%d (%.2f).jpg' % (dir_path, item_idx, item, conf), sub_image)
            '''
            elif track.last_state == 'CMatch':
                cv2.imwrite('%s/%d_%d (%.2f)(C).jpg' % (dir_path, item_idx, item, conf), sub_image)
            elif track.last_state == 'Search':
                cv2.imwrite('%s/%d_%d (%.2f)(S).jpg' % (dir_path, item_idx, item, conf), sub_image)
            '''


def test_cluster_and_outlier_removal():
    print('Loading data ...')
    dump_list = pickle.load(open('temp/result.pkl', 'rb'))
    do_data_cleaning(dump_list)


def do_data_cleaning(data_list, type='body+face', target_state='Match'):
    expand_w = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 4.0]
    expand_h = [1.5, 1.8, 2.0]
    display_mod = False
    overlap_trheshold = 0.1
    save_dir = '/media/data/PTMP/HumanCare/SELF/Exercise'

    # Init Marking
    for idx, dp in enumerate(data_list):
        tracks = dp['Tracks']
        for track_idx, t in enumerate(tracks):
            t.is_valid = False
            if t.type == 'body' and t.face is not None:
                t.face.is_valid = False

    if 'body' in type:
        bc, bi, b_cf = do_cluster_and_outlier_removal(data_list, target_type='body', image_size=(64, 128),
                                                      cluster_ratio=0.2, min_avg_conf=0.5,
                                                      node_ratio=0.3, min_iter=3, max_iter=100,
                                                      target_state=target_state)
        if bc is not None:
            for c in bc:
                for i in range(len(c.index)):
                    seq_idx = c.seq_index[i]
                    track_idx = c.track_index[i]
                    is_outlier = c.outlier_index[i] is 1
                    t = data_list[seq_idx]['Tracks'][track_idx]
                    t.is_valid = not is_outlier

    if 'face' in type:
        fc, fi, f_cf = do_cluster_and_outlier_removal(data_list, target_type='face', image_size=(64, 64),
                                                      cluster_ratio=0.1, min_avg_conf=0.05,
                                                      node_ratio=0.5, min_iter=3, max_iter=100,
                                                      target_state='Search')
        if fc is not None:
            for c in fc:
                for i in range(len(c.index)):
                    seq_idx = c.seq_index[i]
                    track_idx = c.track_index[i]
                    is_outlier = c.outlier_index[i] is 1
                    t = data_list[seq_idx]['Tracks'][track_idx]
                    t.face.is_valid = not is_outlier

    total_count = 0
    valid_count = 0
    for idx, dp in enumerate(data_list):
        img = dp['Image']
        tracks = dp['Tracks']

        disp = img.copy()

        total_count += len(tracks)
        for track_idx, t in enumerate(tracks):
            if t.last_state is 'Search':
                continue

            is_valid = False
            if 'body' in type:
                is_valid = t.is_valid
                if is_valid and 'face' in type:
                    f = t.face
                    if f is None or not f.is_valid:
                        is_valid = False

            # Expand and Crop
            if is_valid and overlap_trheshold > 0:
                body_list = []
                face_list = []

                def get_overlap_count(n_rect):
                    count = 0
                    for t2 in tracks:
                        t_rect = (t2.tl[0], t2.tl[1], t2.br[0], t2.br[1])
                        if util.getOverlapRatio(t_rect, n_rect) > 0:
                            count += 1
                    return count

                def find_maximum_rect(rect):
                    min_count = 0
                    max_rect = None
                    for w_e in expand_w:
                        for h_e in expand_h:
                            n_rect = util.expand_rect(img, rect, w_e, h_e)
                            n_count = get_overlap_count(n_rect)

                            if max_rect is None or n_count <= min_count:
                                min_count = n_count
                                max_rect = n_rect

                    return max_rect

                nx, ny, nw, nh = find_maximum_rect(t.rect)
                n_rect = (nx, ny, nx + nw, ny + nh)
                sub_img = util.crop(img, (nx, ny, nw, nh))

                body_list.append(t.rect)
                if t.face is not None and ('face' in type or '(f)' in type):
                    face_list.append(t.face.rect)

                for track_idx2, t2 in enumerate(tracks):
                    if track_idx2 == track_idx:
                        continue

                    if 'body' in type:
                        t_rect = (t2.tl[0], t2.tl[1], t2.br[0], t2.br[1])
                        if util.is_in_rect(t_rect, n_rect) or util.getOverlapRatio(t_rect, n_rect) > overlap_trheshold:
                            body_list.append(t2.rect)
                            if '(f)' in type and t2.face is not None:
                                face_list.append(t2.face.rect)

                    if 'face' in type and t2.face is not None:
                        t_rect = (t2.face.tl[0], t2.face.tl[1], t2.face.br[0], t2.face.br[1])
                        if util.is_in_rect(t_rect, n_rect) or util.getOverlapRatio(t_rect, n_rect) > 0:
                            face_list.append(t2.face.rect)

                disp = sub_img.copy()
                for idx, r in enumerate(body_list):
                    rx = r[0] - nx
                    ry = r[1] - ny
                    rw = r[2]
                    rh = r[3]

                    if rx + rw > nx + nw:
                        rw = (nx + nw) - rx
                    if ry + rh > ny + nh:
                        rh = (ny + nh) - ry

                    body_list[idx] = (rx, ry, rx + rw, ry + rh)
                    cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 1)

                for idx, r in enumerate(face_list):
                    rx = r[0] - nx
                    ry = r[1] - ny
                    rw = r[2]
                    rh = r[3]

                    if rx + rw > nx + nw:
                        rw = (nx + nw) - rx
                    if ry + rh > ny + nh:
                        rh = (ny + nh) - ry

                    face_list[idx] = (rx, ry, rx + rw, ry + rh)
                    cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)

                # cv2.imshow('Result', disp)
                # cv2.waitKey(50)
                valid_count += 1
                save_annotations(save_dir, sub_img, body_list, face_list)

            if display_mod:
                if t.is_valid:
                    cv2.rectangle(disp, t.tl, t.br, (0, 255, 0), 1)
                else:
                    cv2.rectangle(disp, t.tl, t.br, (0, 0, 255), 1)

                if t.type == 'body' and t.face is not None:
                    f = t.face
                    if f.is_valid:
                        cv2.rectangle(disp, f.tl, f.br, (0, 255, 0), 1)
                    else:
                        cv2.rectangle(disp, f.tl, f.br, (0, 0, 255), 1)

        if display_mod:
            cv2.imshow('Result', disp)
            cv2.waitKey(15)

    if total_count > 0 and valid_count > 0:
        print('%d of %d valid tracks (%.1f%%)' % (valid_count, total_count, (valid_count * 100) / total_count))
    else:
        print("No valid tracks in %d" % (total_count))


def do_cluster_and_outlier_removal(data_list, target_type, image_size=(32, 32), cluster_ratio=0.2,
                                   min_avg_conf=0.7, node_ratio=0.3, min_iter=0, max_iter=100,
                                   target_state='Match'):
    clusters, images, confs = do_clustering(data_list, target_type=target_type,
                                            img_width=image_size[0], img_height=image_size[1],
                                            cluster_ratio=cluster_ratio,
                                            min_avg_conf=min_avg_conf,
                                            target_state=target_state, use_DeepFeature=True)

    if clusters is None:
        return None, None, None

    for cluster_idx, cluster in enumerate(clusters):
        cluster_images = np.zeros(shape=[cluster.num_examples, images.shape[1], images.shape[2], images.shape[3]])
        for i in range(len(cluster.index)):
            idx = cluster.index[i]
            cluster_images[i] = images[idx]

        outlier_index, recon_err = do_outlier_removal(cluster_images, node_ratio=node_ratio, min_iter=min_iter,
                                                      max_iter=max_iter)
        cluster.outlier_index = outlier_index
        cluster.valid_count = len(outlier_index) - np.count_nonzero(outlier_index)

    return clusters, images, confs


def benchmark_cluster_kmeans():
    data = pickle.load(open('temp/tracking_dump_0.pickle', 'rb'))
    w = 80
    h = 160

    images = []
    confs = []
    labels = []

    for data_idx in [0, 1]:
        print('Count: %d' % len(data[data_idx]))
        for i in range(len(data[data_idx])):
            img = cv2.resize(data[data_idx][i][0], (w, h))
            img = np.reshape(img, (w * h * 3))
            images.append(img)
            labels.append(data_idx)
            confs.append(data[data_idx][i][1])

    avg_purities = []
    for exp_idx in range(10):
        clusters = Grouping.do_kMeans(10, images, confs, labels)

        cluster_confs = []
        for c in clusters:
            cluster_confs.append(c.average_confidence)

        cluster_rank = np.argsort(cluster_confs)[::-1]
        cluster_purity_sum = 0
        for i in range(7):
            cluster_purity_sum += clusters[cluster_rank[i]].purity
        cluster_purity_sum /= 7
        print(cluster_purity_sum)
        avg_purities.append(cluster_purity_sum)

    print('Avg.Purity : %.4f (Std: %.4f)' % (np.mean(avg_purities), np.std(avg_purities)))


def benchmark_cluster_ROC():
    data = pickle.load(open('temp/tracking_dump_0.pickle', 'rb'))
    w = 25
    h = 50

    images = []
    confs = []
    labels = []

    for data_idx in [0, 1]:
        print('Count: %d' % len(data[data_idx]))
        for i in range(len(data[data_idx])):
            img = cv2.resize(data[data_idx][i][0], (w, h))
            img = np.reshape(img, (w * h * 3))
            images.append(img)
            labels.append(data_idx)
            confs.append(data[data_idx][i][1])

    images = np.array(images)
    clusters = Grouping.do_ROClustering(15, 10, 0, images, confs, labels, None)

    cluster_confs = []
    for c in clusters:
        cluster_confs.append(c.average_confidence)

    cluster_rank = np.argsort(cluster_confs)[::-1]
    cluster_purity_sum = 0
    for i in range(7):
        c = clusters[cluster_rank[i]]
        print('[%d] %d images, purity %.2f' % (i, c.num_examples, c.purity))
        cluster_purity_sum += c.purity
    cluster_purity_sum /= 7
    print('Avg.Purity : %.4f' % cluster_purity_sum)


def benchmark_cluster_ROC_AE():
    data = pickle.load(open('temp/tracking_dump_0.pickle', 'rb'))
    w = 5
    h = 10

    images = []
    confs = []
    labels = []

    for data_idx in [0, 1]:
        print('Count: %d' % len(data[data_idx]))
        for i in range(len(data[data_idx])):
            img = cv2.resize(data[data_idx][i][0], (w, h))
            img = np.reshape(img, (w * h * 3))
            images.append(img)
            labels.append(data_idx)
            confs.append(data[data_idx][i][1])

    images = np.array(images)

    clusters = Grouping.do_ROClustering(15, 10, 0, images, confs, labels, None)
    cluster_confs = []
    for c in clusters:
        cluster_confs.append(c.average_confidence)

    cluster_rank = np.argsort(cluster_confs)[::-1]
    cluster_purity_sum = 0
    for i in range(7):
        c = clusters[cluster_rank[i]]
        print('[%d] %d images, purity %.2f' % (i, c.num_examples, c.purity))
        cluster_purity_sum += c.purity
    cluster_purity_sum /= 7
    print('Avg.Purity : %.4f' % cluster_purity_sum)

    print('--------------------------------------------')

    for cluster_idx, c in enumerate(clusters):
        cluster_images = images[c.index]
        out_idx, out_err = Grouping.do_OutlierRemove(0.05, 5, 10, cluster_images)
        new_cluster = Grouping.GroupInfo()
        for idx, is_major in enumerate(out_idx):
            if is_major == 0:
                new_cluster.index.append(c.index[idx])
                new_cluster.confidence.append(c.confidence[idx])
                new_cluster.labels.append(c.labels[idx])
        clusters[cluster_idx] = new_cluster

    cluster_purity_sum = 0
    for i in range(7):
        c = clusters[cluster_rank[i]]
        print('[%d] %d images, purity %.2f' % (i, c.num_examples, c.purity))
        cluster_purity_sum += c.purity
    cluster_purity_sum /= 7
    print('Avg.Purity : %.4f' % cluster_purity_sum)


def correct_annotation_files():
    dir = '/home/youbit/storage/PTMP/HumanCare/VOC2012R/Train'
    annotation_dir = dir + '/Annotations'
    img_dir = dir + '/Images'
    show_width = 600

    task = 'B-P'
    file_list = os.listdir(annotation_dir)

    def getRect(obj):
        bndbox = obj.find('bndbox')
        x1 = bndbox.find('xmin').text
        x2 = bndbox.find('xmax').text
        y1 = bndbox.find('ymin').text
        y2 = bndbox.find('ymax').text

        if '.' in x1 or ',' in x2 or '.' in y1 or '.' in y2:
            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))

            bndbox.find('xmin').text = str(x1)
            bndbox.find('xmax').text = str(x2)
            bndbox.find('ymin').text = str(y1)
            bndbox.find('ymax').text = str(y2)

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return (x1, y1, x2, y2)

    for idx, filePath in enumerate(file_list):
        tree = XMLTree.parse('%s/%s' % (annotation_dir, filePath))
        root = tree.getroot()

        if idx % 1000 == 0:
            print('Checking ... %d' % idx)

        is_modified = False

        # Name Change : Person to Body
        if task == 'P-B':
            objs = root.findall('object')

            for obj in objs:
                obj_name = obj.find('name')
                if obj_name.text == 'person':
                    obj_name.text = 'body'
                    is_modified = True

        # Name Change : Body to Person
        elif task == 'B-P':
            objs = root.findall('object')
            for obj in objs:
                obj_name = obj.find('name')
                if obj_name.text == 'body':
                    obj_name.text = 'person'
                    is_modified = True

        # Coordinate Validation
        elif task == 'VA':
            objs = root.findall('object')
            for obj in objs:
                bndbox = obj.find('bndbox')
                x1, y1, x2, y2 = getRect(obj)
                if x1 >= x2:
                    bndbox.find('xmin').text = str(x2)
                    bndbox.find('xmax').text = str(x1)
                    print('%s - x:%d/%d' % (filePath, x1, x2))
                    return

                if y1 >= y2:
                    bndbox.find('ymin').text = str(y2)
                    bndbox.find('ymax').text = str(y1)
                    print('%s - y:%d/%d' % (filePath, y1, y2))
                    return

                img_path = img_dir + '/' + root.find('filename').text
                if not os.path.exists(img_path):
                    print('%s - %s is not exist!' % (filePath, img_path))
                    return
                else:
                    img = cv2.imread(img_path)
                    if img is None:
                        print('%s - %s is not valid!' % (filePath, img_path))
                        return

        # Visualization
        elif task == 'Show':
            filename = root.find('filename').text
            img = cv2.imread(img_dir + '/' + filename)

            resize_rate = show_width / img.shape[1]
            img = cv2.resize(img, (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate)))

            objs = root.findall('object')
            for obj in objs:
                x1, y1, x2, y2 = getRect(obj)

                x1 = int(x1 * resize_rate)
                x2 = int(x2 * resize_rate)
                y1 = int(y1 * resize_rate)
                y2 = int(y2 * resize_rate)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow('Image', img)
            cv2.waitKey()

        # Separate body to sub regions
        elif task == 'Separate':
            filename = root.find('filename').text
            img = cv2.imread(img_dir + '/' + filename)

            objs = root.findall('object')
            class BodyRegion:
                pass

            body_list = []

            # Make body region object
            for obj in objs:
                obj_name = obj.find('name').text
                x1, y1, x2, y2 = getRect(obj)

                if obj_name in ['body', 'person'] :
                    b = BodyRegion()
                    b.name = obj_name
                    b.tl = (x1, y1)
                    b.br = (x2, y2)
                    b.rect = (x1, y1, x2 - x1, y2 - y1)
                    body_list.append(b)

                # Remove exist annotation
                elif obj_name in ['body(u)', 'body(b)']:
                    root.remove(obj)

            # Find corresponding face
            def find_face(b):
                best_face = None
                best_overlap = 0
                for obj in objs:
                    obj_name = obj.find('name').text
                    x1, y1, x2, y2 = getRect(obj)

                    if obj_name in ['face']:
                        r = (x1, y1, x2 - x1, y2 - y1)
                        overlap =  util.getOverlapRatio(r, b.rect, is_rect=True)
                        if overlap > best_overlap:
                            best_face = r
                            best_overlap = overlap

                return best_face

            for b in body_list:
                b.face = find_face(b)

            # Split Region
            for b in body_list:
                b.upper = None
                b.bottom = None

                cv2.rectangle(img, b.tl, b.br, (255, 255, 255), 1)
                if b.face is not None:
                    cv2.rectangle(img, util.rect_to_point(b.face)[0], util.rect_to_point(b.face)[1], (255, 255, 255), 1)

                if b.rect[2] <= 0 or b.rect[3] <= 0:
                    continue

                wh_ratio = b.rect[2] / b.rect[3]
                if wh_ratio < 1:
                    upper_rect = (b.rect[0], b.rect[1], b.rect[2], int(b.rect[3] / 2))
                    bottom_rect = (b.rect[0], b.rect[1] + int(b.rect[3] / 2), b.rect[2], int(b.rect[3] / 2))

                    if b.face is not None and util.getOverlapRatio(bottom_rect, b.face, is_rect=True) > 0:
                        continue
                    else:
                        b.upper = upper_rect
                        b.bottom = bottom_rect

                        n = XMLTree.Element('object')
                        XMLTree.SubElement(n, 'name').text = 'body(u)'

                        bndbox = XMLTree.Element('bndbox')
                        XMLTree.SubElement(bndbox, 'xmin').text = str(b.upper[0])
                        XMLTree.SubElement(bndbox, 'ymin').text = str(b.upper[1])
                        XMLTree.SubElement(bndbox, 'xmax').text = str(b.upper[0] + b.upper[2])
                        XMLTree.SubElement(bndbox, 'ymax').text = str(b.upper[1] + b.upper[3])
                        n.append(bndbox)
                        root.append(n)

                        n = XMLTree.Element('object')
                        XMLTree.SubElement(n, 'name').text = 'body(b)'

                        bndbox = XMLTree.Element('bndbox')
                        XMLTree.SubElement(bndbox, 'xmin').text = str(b.bottom[0])
                        XMLTree.SubElement(bndbox, 'ymin').text = str(b.bottom[1])
                        XMLTree.SubElement(bndbox, 'xmax').text = str(b.bottom[0] + b.bottom[2])
                        XMLTree.SubElement(bndbox, 'ymax').text = str(b.bottom[1] + b.bottom[3])
                        n.append(bndbox)
                        root.append(n)

                        is_modified = True

                if b.upper is not None :
                    cv2.rectangle(img, util.rect_to_point(b.upper)[0], util.rect_to_point(b.upper)[1], (0, 0, 255), 1)
                    cv2.rectangle(img, util.rect_to_point(b.bottom)[0], util.rect_to_point(b.bottom)[1], (255, 0, 0), 1)

            # cv2.imshow('Result', img)
            # cv2.waitKey()

        else:
            raise Exception('Invalid task = ' + task)


        if is_modified:
            print(filePath)
            util.indentXML(root)
            XMLTree.ElementTree(root).write('%s/%s' % (annotation_dir, filePath))


def convert_annotation_files():
    dir = '/home/youbit/storage/Project/HumanCare/Dataset/Sitting+VOC+Walking'
    target_file = '/home/youbit/storage/Project/HumanCare/Workspaces/YOLOv3/model/person_train.txt'
    annotation_dir = dir + '/Annotations'
    file_list = os.listdir(annotation_dir)
    class_names = ['person', 'face', 'body(u)', 'body(b)']
    class_count = [0] * 4

    def getRect(obj):
        bndbox = obj.find('bndbox')
        x1 = bndbox.find('xmin').text
        x2 = bndbox.find('xmax').text
        y1 = bndbox.find('ymin').text
        y2 = bndbox.find('ymax').text

        if '.' in x1 or ',' in x2 or '.' in y1 or '.' in y2:
            x1 = int(float(x1))
            y1 = int(float(y1))
            x2 = int(float(x2))
            y2 = int(float(y2))

            bndbox.find('xmin').text = str(x1)
            bndbox.find('xmax').text = str(x2)
            bndbox.find('ymin').text = str(y1)
            bndbox.find('ymax').text = str(y2)

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        return x1, y1, x2, y2

    wt = open(target_file, 'w')
    for idx, filePath in enumerate(file_list):
        tree = XMLTree.parse('%s/%s' % (annotation_dir, filePath))
        root = tree.getroot()

        if idx % 1000 == 0:
            print('Export annotation for yolo3 ... %d' % idx)

        filename = root.find('filename').text
        img_file_path = '%s/Images/%s' % (dir, filename)
        if not os.path.exists(img_file_path):
            raise Exception('[%s] is not exist!' % img_file_path)
        wt.write(img_file_path)

        objs = root.findall('object')
        for obj in objs:
            obj_name = obj.find('name').text

            if obj_name == 'body':
                obj_name = 'person'
            if obj_name not in class_names:
                continue

            cls_idx = class_names.index(obj_name)
            x1, y1, x2, y2 = getRect(obj)
            wt.write(' %d,%d,%d,%d,%d' % (x1, y1, x2, y2, cls_idx))

            class_count[cls_idx] += 1

        wt.write('\n')

    for i in range(4):
        print('%s : %d' % (class_names[i], class_count[i]))
    wt.close()


pt1 = None
pt2 = None
pt_cur = None


def do_body_annotation():
    dir = '/home/youbit/Download/Google Images/2017-10-27 11-7-51'
    save_dir = '/media/data/PTMP/HumanCare/Custom_Sitting'

    body_list = []
    face_list = []

    global pt1, pt2
    pt1 = None
    pt2 = None
    current_type = 0

    task = ''
    if task != 'Show':
        def mouse_callback(event, x, y, flags, param):
            global pt1, pt2, pt_cur

            if event == cv2.EVENT_LBUTTONDOWN:
                if pt1 is None:
                    pt1 = (x, y)
                else:
                    if x > pt1[0] and y > pt1[1]:
                        pt2 = (x, y)
                        if current_type == 0:
                            body_list.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                        else:
                            face_list.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                    pt1 = None
                    pt2 = None

            elif event == cv2.EVENT_MOUSEMOVE:
                pt_cur = (x, y)
                if pt1 is not None:
                    pt2 = (x, y)

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', mouse_callback)

        do_detection = True
        delete_after_done = True

        if do_detection:
            options = {"model": "cfg/yolo-b.cfg", "load": 10000, "threshold": 0.5, 'gpu': 0.0}
            # options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.0, 'gpu': 2.0}
            tfnet = TFNet(options)

        for fileName in os.listdir(dir):
            body_list = []
            face_list = []
            current_type = 0

            image_path = '%s/%s' % (dir, fileName)
            img = cv2.imread(image_path)

            if img is None:
                continue

            resizeRate = 600 / img.shape[0]
            resized = cv2.resize(img, None, fx=resizeRate, fy=resizeRate)

            if do_detection:
                result = tfnet.return_predict(resized)
                for r in result:
                    cls_name = r['label']
                    rect = [r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'], r['bottomright']['y']]

                    if cls_name == 'body' or cls_name == 'person':
                        body_list.append(rect)
                    elif cls_name == 'face':
                        face_list.append(rect)

            if img is None or img.shape[0] == 0:
                continue

            while True:
                disp = resized.copy()

                for b in body_list:
                    cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)

                for f in face_list:
                    cv2.rectangle(disp, (f[0], f[1]), (f[2], f[3]), (0, 0, 255), 1)

                if pt1 is not None and pt2 is not None:
                    cv2.rectangle(disp, pt1, pt2, (0, 255, 255), 1)

                if pt_cur is not None:
                    cv2.line(disp, (pt_cur[0], 0), (pt_cur[0], disp.shape[0]), (0, 200, 200), 1)
                    cv2.line(disp, (0, pt_cur[1]), (disp.shape[1], pt_cur[1]), (0, 200, 200), 1)

                cv2.imshow('Image', disp)
                key = cv2.waitKey(10)

                # Cancel
                if key == 27:
                    if delete_after_done:
                        os.remove(image_path)
                    break

                # Continue
                elif key == 32:
                    if len(body_list) + len(face_list) > 0:
                        for b in body_list:
                            for i in range(len(b)):
                                b[i] = round(b[i] / resizeRate)
                        for f in face_list:
                            for i in range(len(f)):
                                f[i] = round(f[i] / resizeRate)
                        save_annotations(save_dir, img, body_list, face_list)

                        if delete_after_done:
                            os.remove(image_path)
                    break

                elif key == ord('r'):
                    if current_type == 0:
                        if len(body_list) > 0:
                            del body_list[-1]
                    elif current_type == 1:
                        if len(face_list) > 0:
                            del face_list[-1]

                elif key == ord('1'):
                    current_type = 0
                    print('Current Type : Body')
                elif key == ord('2'):
                    current_type = 1
                    print('Current Type : Face')
                elif key == ord('5'):
                    body_list = []
                elif key == ord('6'):
                    face_list = []

    else:
        for fileName in os.listdir(save_dir + '/Images'):
            img = cv2.imread(save_dir + '/Images/' + fileName)
            tree = XMLTree.parse(save_dir + '/Annotations/' + fileName[0:-4] + ".xml")
            root = tree.getroot()
            objs = root.findall('object')

            for obj in objs:
                obj_name = obj.find('name')
                bndbox = obj.find('bndbox')
                x1 = int(bndbox.find('xmin').text)
                x2 = int(bndbox.find('xmax').text)
                y1 = int(bndbox.find('ymin').text)
                y2 = int(bndbox.find('ymax').text)

                if obj_name.text == 'body':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                elif obj_name.text == 'face':
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.imshow('Image', img)
            cv2.waitKey()


def do_INRIA_annotation():
    dir = '/media/data/PTMP/HumanCare/Custom_INRIAPerson'

    global pt1, pt2
    pt1 = None
    pt2 = None
    current_type = 0

    face_list = []

    def mouse_callback(event, x, y, flags, param):
        global pt1, pt2, pt_cur

        if event == cv2.EVENT_LBUTTONDOWN:
            if pt1 is None:
                pt1 = (x, y)
            else:
                pt2 = (x, y)
                if current_type == 0:
                    body_list.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                else:
                    face_list.append([pt1[0], pt1[1], pt2[0], pt2[1]])
                pt1 = None
                pt2 = None

        elif event == cv2.EVENT_MOUSEMOVE:
            pt_cur = (x, y)
            if pt1 is not None:
                pt2 = (x, y)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    fileNames = os.listdir(dir + '/Images')
    fileNames.sort()
    for idx, fileName in enumerate(fileNames):
        img = cv2.imread(dir + '/Images/' + fileName)

        fileName = fileName[0:-4]
        tree = XMLTree.parse(dir + '/Annotations/' + fileName + ".xml")
        root = tree.getroot()

        if root.find('folder').text == 'INRIA-Custom':
            print('%d:%s - PASS' % (idx + 1, fileName))
            continue

        print('%d:%s' % (idx + 1, fileName))
        objs = root.findall('object')

        face_list = []
        body_list = []
        current_type = 0

        resizeRate = 600 / img.shape[0]
        resized = cv2.resize(img, None, fx=resizeRate, fy=resizeRate)

        for obj in objs:
            bndbox = obj.find('bndbox')
            x1 = int(int(bndbox.find('xmin').text) * resizeRate)
            x2 = int(int(bndbox.find('xmax').text) * resizeRate)
            y1 = int(int(bndbox.find('ymin').text) * resizeRate)
            y2 = int(int(bndbox.find('ymax').text) * resizeRate)

            obj_name = obj.find('name').text
            if obj_name == 'body' or obj_name == 'person':
                body_list.append([x1, y1, x2, y2])
            elif obj_name == 'face':
                face_list.append([x1, y1, x2, y2])

        while True:
            disp = resized.copy()

            for f in face_list:
                cv2.rectangle(disp, (f[0], f[1]), (f[2], f[3]), (0, 0, 255), 1)

            for b in body_list:
                cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)

            if pt1 is not None and pt2 is not None:
                cv2.rectangle(disp, pt1, pt2, (0, 255, 255), 1)

            if pt_cur is not None:
                cv2.line(disp, (pt_cur[0], 0), (pt_cur[0], disp.shape[0]), (0, 200, 200), 1)
                cv2.line(disp, (0, pt_cur[1]), (disp.shape[1], pt_cur[1]), (0, 200, 200), 1)

            cv2.imshow('Image', disp)
            key = cv2.waitKey(10)

            # Cancel
            if key == 27:
                break

            # Continue
            elif key == 32:
                for f in face_list:
                    for i in range(len(f)):
                        f[i] = round(f[i] / resizeRate)
                for b in body_list:
                    for i in range(len(b)):
                        b[i] = round(b[i] / resizeRate)
                save_annotations(dir, img, body_list, face_list, filename=fileName)
                break

            elif key == ord('r'):
                if current_type == 0:
                    if len(body_list) > 0:
                        del body_list[-1]
                elif current_type == 1:
                    if len(face_list) > 0:
                        del face_list[-1]

            elif key == ord('1'):
                current_type = 0
                print('Current Type : Body')
            elif key == ord('2'):
                current_type = 1
                print('Current Type : Face')


def do_expression_annotation():
    dir = '/media/data/PTMP/HumanCare/YLF'
    for fileName in os.listdir(dir + '/Images'):
        img = cv2.imread(dir + '/Images/' + fileName)
        tree = XMLTree.parse(dir + '/Annotations/' + fileName[0:-4] + ".xml")
        root = tree.getroot()

        objs = root.findall('object')

        face_count = 0
        face_obj = None
        for obj in objs:
            obj_name = obj.find('name')
            bndbox = obj.find('bndbox')
            x1 = int(bndbox.find('xmin').text)
            x2 = int(bndbox.find('xmax').text)
            y1 = int(bndbox.find('ymin').text)
            y2 = int(bndbox.find('ymax').text)

            if obj_name.text == 'body':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            elif obj_name.text == 'face':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                face_count += 1
                face_obj = obj

        if face_count == 1:

            ra_node = face_obj.find('Reaction')
            if ra_node is not None:
                continue

            reaction = 'Neutral'
            is_focused = False
            is_talking = False
            question_no = 0

            while True:

                disp = img.copy()

                txt = reaction
                if is_focused:
                    txt += '/Focused/'
                else:
                    txt += '/Not Focused/'

                if is_talking:
                    txt += 'Talking'
                else:
                    txt += 'Not Talking'

                cv2.rectangle(disp, (10, 10), (500, 50), (0, 0, 0), -1)
                cv2.putText(disp, txt, (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                cv2.imshow('Image', disp)

                key = cv2.waitKey(10)

                if key == ord('r'):
                    question_no = 0

                # Pass
                elif key == 27:
                    break

                # Save
                elif key == 32:

                    ra_node = face_obj.find('Reaction')
                    fc_node = face_obj.find('Focusing')
                    tk_node = face_obj.find('Talking')

                    if ra_node is not None:
                        ra_node.text = reaction
                    else:
                        XMLTree.SubElement(face_obj, 'Reaction').text = reaction

                    if fc_node is not None:
                        fc_node.text = str(is_focused)
                    else:
                        XMLTree.SubElement(face_obj, 'Focusing').text = str(is_focused)

                    if tk_node is not None:
                        tk_node.text = str(is_talking)
                    else:
                        XMLTree.SubElement(face_obj, 'Talking').text = str(is_talking)

                    util.indentXML(root)
                    XMLTree.ElementTree(root).write(dir + '/Annotations/' + fileName[0:-4] + ".xml")
                    break

                if key == ord('1'):
                    if question_no == 0:
                        reaction = 'Positive'
                    elif question_no == 1:
                        is_focused = True
                    elif question_no == 2:
                        is_talking = True
                    question_no += 1
                    if question_no > 2:
                        question_no = 0

                elif key == ord('2'):
                    if question_no == 0:
                        reaction = 'Negative'
                    elif question_no == 1:
                        is_focused = False
                    elif question_no == 2:
                        is_talking = False
                    question_no += 1
                    if question_no > 2:
                        question_no = 0

                elif key == ord('3'):
                    if question_no == 0:
                        reaction = 'Neutral'
                        question_no += 1


def do_save_VOC():
    dir = '/media/data/PTMP/HumanCare/VOC_Person+Face'
    save_dir = '/media/data/PTMP/HumanCare/VOC_Person'
    for fileName in os.listdir(dir + '/Images'):
        img = cv2.imread(dir + '/Images/' + fileName)
        tree = XMLTree.parse(dir + '/Annotations/' + fileName[0:-4] + ".xml")
        root = tree.getroot()

        objs = root.findall('object')

        is_valid = False
        for obj in objs:
            name = obj.find('name')
            if name.text == 'person':
                # name.text = 'person'
                is_valid = True
                break

        if is_valid:
            print(fileName)
            cv2.imwrite('%s/Images/%s' % (save_dir, fileName), img)
            util.indentXML(root)
            XMLTree.ElementTree(root).write('%s/Annotations/%s' % (save_dir, fileName[0:-4] + ".xml"))


def do_save_YLF():
    dir = '/media/data/PTMP/HumanCare/YLF'
    save_dir = '/media/data/PTMP/HumanCare/Custom_YLF'
    for fileName in os.listdir(dir + '/Images'):
        img = cv2.imread(dir + '/Images/' + fileName)
        tree = XMLTree.parse(dir + '/Annotations/' + fileName[0:-4] + ".xml")
        root = tree.getroot()

        objs = root.findall('object')

        body_count = 0
        for obj in objs:
            obj_name = obj.find('name').text

            if obj_name == 'body' or obj_name == 'person':
                bndbox = obj.find('bndbox')
                x1 = int(bndbox.find('xmin').text)
                x2 = int(bndbox.find('xmax').text)
                y1 = int(bndbox.find('ymin').text)
                y2 = int(bndbox.find('ymax').text)

                w = x2 - x1
                h = y2 - y1

                if (w * h) / (img.shape[0] * img.shape[1]) < 0.2:
                    body_count += 1
                else:
                    body_count = 0
                    break

        if body_count > 0:
            print(fileName)
            cv2.imwrite('%s/Images/%s' % (save_dir, fileName), img)
            util.indentXML(root)
            XMLTree.ElementTree(root).write('%s/Annotations/%s' % (save_dir, fileName[0:-4] + ".xml"))


def do_collection_by_dection():
    src_path = '/home/youbit/Download/Google Images/2017-10-26 20-26-44'
    dst_path = '/media/data/PTMP/HumanCare/SELF_YOLO'

    dp = DataProvider.ImageDataProvider(src_path)

    # options = {"model": "cfg/yolo-a.cfg", "load": 1900, "threshold": 0.0, 'gpu': 2.0}
    options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.8, 'gpu': 2.0}
    tfnet = TFNet(options)

    while True:
        img = dp.get()
        disp = img.copy()
        if img is None:
            break

        result = tfnet.return_predict(img)

        if (len(result) != 1):
            continue

        body_list = []
        face_list = []
        for r in result:
            if not r['label'] in ['person', 'body']:
                continue

            rect = (r['topleft']['x'], r['topleft']['y'], r['bottomright']['x'], r['bottomright']['y'])
            body_list.append(rect)
            cv2.rectangle(disp, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 1)

        cv2.imshow('Image', disp)
        cv2.waitKey()

        if len(body_list) > 0 or len(face_list) > 0:
            save_annotations(dst_path, img, body_list, face_list)

def train_and_test_background():
    model_file = 'temp/background_model/model-%d'
    model_no = 0
    img_size = (64, 64)
    img_channel = 1
    bg_mask_size = (20, 40, 20, 40)

    bg_dp = DataProvider.BackgroundImageDataProvider(src_dir='/media/data/ETRI-HumanCare/LivingLabBG',
                                                     img_size=img_size, channel=img_channel, mask_size=bg_mask_size,
                                                     label_type='Flat')

    is_train = False
    is_elm = True

    # Training
    if is_train:
        ds = DataInput.DataSets()
        ds.train = bg_dp

        ds.train.batch_manager.cache_dir = '/media/data/temp'
        ds.train.batch_manager.cache_prefix = 'background-'
        ds.train.batch_manager.cache_refresh = 10
        ds.train.batch_manager.cache_mem_interval = 1
        ds.train.batch_manager.cache_hit_prob = 0.9
        ds.train.batch_manager.delete_cache_files()

        aug_params = util.ImageAugmentationParams()
        aug_params.rotate = 0
        aug_params.crop_area_min = 0.9
        aug_params.crop_area_max = 1.0
        aug_params.bright_delta = 30

        ds.train.batch_manager.set_augmentation(aug_params)
        ds.train.init()

        if '' == 'D':
            ds.train.show_data(100)

        print(ds.train.num_examples, 'Samples ...')

        ELMCNN.OPTION_CONVERGE_LIMIT = 0
        ELMCNN.OPTION_SGD_PRINT_UNIT = 'Epoch/Batch'
        ELMCNN.OPTION_SGD_AUTOSAVE_TERM = 100
        ELMCNN.OPTION_SGD_START_EPOCH = model_no
        ELMCNN.OPTION_SGD_RECORD_COST = 'temp/cost_history.csv'
        ELMCNN.OPTION_SGD_RECORD_APPEND = True
        ELMCNN.OPTION_AUTO_CREATE_IMPLICIT_DATA_AND_LAYER = False

        if ELMCNN.OPTION_SGD_RECORD_COST != '' and ELMCNN.OPTION_SGD_RECORD_APPEND and \
                os.path.exists(ELMCNN.OPTION_SGD_RECORD_COST) and model_no == 0:
            os.remove(ELMCNN.OPTION_SGD_RECORD_COST)

        net_strings = [
            'C(3/3/16/R/OG/S/BN)-ReLU-MP(2/2)-'
            'C(3/3/16/R/OG/S/BN)-ReLU-MP(2/2)-'
            # 'FT(10/1e-5/F/1000)-'
            'Flatten-'            
            # 'FC(2048)-ReLU-'
            # 'FC(_LABEL_DIM_)'
            'Sample(20000)-'
            'ELM(4096/S/Sig)'
        ]

        for net_str in net_strings:
            net_info = ELMCNN.parse_net_string(net_str, params=[], is_trainable=True, dataset=ds.train),
            net = net_info[0].net

            ELMCNN.calc_layer_params(ds, net)
            net_name = ELMCNN.get_network_architecture(net)
            print(net_name)

            load_model = ''
            if model_no > 0:
                load_model = model_file % model_no

            cl = ELMCNN()
            cl.batch_size = 8
            cl.verbose = True
            cl.net_str = net_name

            if is_elm:
                cl.train(ds, net)
                cl.save(model_file % 0)
            else:
                cl.train_sgd(ds, net, n_epoch=500, learning_rate=['Cyclic-T2', 1e-7, 1e-4, 5, 0.9],
                             momentum=0.001, stop_delta=1e-4, bottom_limit=1e-6, display_step=100,
                             loss='RMSE', optimizer='Adam', gradient_clip_value=1e-2,
                             auto_save_model=model_file, load_model=load_model)
    # Test
    else:
        net = TensorflowNetwork('temp/background_model/model-0', 'Inp', 'Out')
        net.load()

        dpType = 2
        dp = None

        if dpType == 1:
            dp = DataProvider.LivingLabDataProvider('/media/data/ETRI-HumanCare/LivingLabSet', 2)

        elif dpType == 2:
            dp = bg_dp

        while True:
            if dpType == 1:
                img = dp.get()
                img = cv2.resize(img, img_size)
                if img_channel == 1 and len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 0)
                img = np.expand_dims(img, 3)
                img = img / 255.0

            elif dpType == 2:
                img, _ = dp.next_batch(1)

            ret = net.feedforward(img)[0][0]

            if len(ret.shape) == 1:
                ret = np.reshape(ret, newshape=(img_size[0], img_size[1], img_channel))

            bg_img = (ret * 255).astype(np.uint8)
            img = (img[0] * 255).astype(np.uint8)
            diff_img = np.average(np.abs(img[0] - bg_img), axis=2)
            # diff_img = diff_img + np.min(diff_img)
            # diff_img[np.where(diff_img < 0.5)] = 0

            cv2.imshow('Input', img)
            cv2.imshow('Output', bg_img)
            cv2.imshow('Diff', diff_img)
            key = cv2.waitKey()
            if key == 27:
                break


load_config('train_and_test.cfg')

# use_opencv_tracker()
# test_tracking()
# record_tracking_result('./temp', 1.0)

# correct_annotation_files()
# convert_annotation_files()
# test_detection()
# train_and_test()

# test_cluster()
# test_outlier_removal('temp/cluster/10 (0.84)')
# test_cluster_and_outlier_removal()
train_and_test_background()

# benchmark_cluster_kmeans()
# benchmark_cluster_ROC()
# benchmark_cluster_ROC_AE()

# do_body_annotation()
# do_INRIA_annotation()
# do_save_VOC()
# do_save_YLF()
# do_expression_annotation()
# do_collection_by_dection()
