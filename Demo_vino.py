import os
import time
import sys
import cv2
import DataProvider
import numpy as np

from calcHistogram import calcHistogram3D
from calcHistogram import calcOrigin

from tracking.TrackMod import TrackMod, DET_THRESHOLD, MIN_DET_THRESHOLD
from tracking.TrackMod_own import TrackMod_own, DET_THRESHOLD, MIN_DET_THRESHOLD


#SAVEBBOX = True

SAVEBBOX = False

#cropping_ratio = (0.8, 0.8)
cropping_ratio = (1.0, 1.0)



origin_resize_ratio = (32, 32)





# feature type: yolo, colorHist

def saveBbox(fileName, img, frameIdx, boxList, resizeFactor=0.4):

    if(len(boxList) == 0):
        return

    fileNameAbsFolderPath = fileName
    if(not os.path.exists(fileNameAbsFolderPath)):
        os.mkdir(fileNameAbsFolderPath)
    frameIdxAbsPath = os.path.join(fileNameAbsFolderPath, str(frameIdx) + '.csv')

    # write bbox info
    with open(frameIdxAbsPath, 'w') as writeFile:
        writeFile.write(str(len(boxList)) + '\n')
        for eachBox in boxList:
            bbox = eachBox[0]
            id = eachBox[1]
            last_state = eachBox[2]
            type = eachBox[3]
            writeFile.write(str(type) + ',' + str(id) + ',' + last_state + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ','+ str(bbox[2]) + ',' + str(bbox[3]) + '\n')
            print(str(type) + ',' + str(id) + ',' + last_state + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ','+ str(bbox[2]) + ',' + str(bbox[3]))

    # write cropped img info
    idx = 0
    for eachBox in boxList:
        bbox = eachBox[0]
        id = eachBox[1]

        x1 = int(bbox[0] / resizeFactor)
        y1 = int(bbox[1] / resizeFactor)
        x2 = int(bbox[2] / resizeFactor)
        y2 = int(bbox[3] / resizeFactor)

        img_Cropped = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(fileNameAbsFolderPath, str(frameIdx) + '_' + str(id) + '.png'), img_Cropped)
        idx+=1

def old_tracker_save_boxed(argv=None):

    # 6 videos for re-calculate the bbox with old box
    #parent_path = '../data/tmp'
    #parent_video_path = '../data/tmp/video'
    #datasetName = 'human_track_data_tmp_kaist-old-tracker'

    # add 2 videos for the certification

    parent_path = '../data/tmp_cert'
    parent_video_path = '../data/tmp_cert/video'
    datasetName = 'human_track_data_tmp_kaist-old-tracker'

    fileList = []

    def loadVideoPathFromParent():
        videoPathList = os.listdir(parent_video_path)

        for eachVideoPath in videoPathList:
            eachVideoAbsPath = os.path.join(parent_video_path, eachVideoPath)

            fileList.append(eachVideoAbsPath)

    loadVideoPathFromParent()

    for eachFile in fileList:
        dp = DataProvider.VideoDataProvider(eachFile)

        resize_rate = 0.4

        '''
        t1 = TrackMod(conf_file="cfg/yolo.cfg", model_file='bin/yolo.weights',
                      det_threshold=0.5, min_det_treshold=0.5,
                      use_cmatch=False, tracker_type='TM', tracker_limit=60)
        '''

        t2 = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=0.2, min_det_treshold=0.1,
                      use_cmatch=True, tracker_type='TM', tracker_limit=10, old_tracker=True)

        frameIdx = 0

        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            # t1.run(img_resize)
            t2.run(img_resize)
            bboxListT2 = []

            for trackRes in t2.current_tracks:
                id = trackRes.id
                x1 = trackRes.tl[0]
                y1 = trackRes.tl[1]
                x2 = trackRes.br[0]
                y2 = trackRes.br[1]

                last_state = trackRes.last_state

                type = trackRes.type

                bbox = (x1, y1, x2, y2)
                bboxListT2.append((bbox, id, last_state, type))

            print('Processing ' + str(frameIdx))

            if SAVEBBOX == True:
                outParentPath = os.path.join(parent_path, datasetName)
                if not os.path.exists(outParentPath):
                    os.mkdir(outParentPath)

                saveBbox(os.path.join(outParentPath, eachFile.split('/')[-1] + '_t2'), img, frameIdx,
                         bboxListT2)

            frameIdx += 1

def own_test_detection(parent_path, dataset_name, feature_type, classifier_type, out_parent_path='tmp'):
    fileList = []

    def loadVideoPathFromParent():
        videoPathList = os.listdir(parent_path)

        for eachVideoPath in videoPathList:
            eachVideoAbsPath = os.path.join(parent_path, eachVideoPath)

            verified_output_path = os.path.join(out_parent_path, dataset_name,
                                                eachVideoPath.split('/')[-1].split('.')[0])

            # exclude the done file
            if os.path.exists(verified_output_path):
                continue

            fileList.append(eachVideoAbsPath)

    loadVideoPathFromParent()

    for eachFile in fileList:
        print('Processing ' + eachFile)
        dp = DataProvider.VideoDataProvider(eachFile)

        resize_rate = 0.4

        t2 = TrackMod_own(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                      min_det_treshold=MIN_DET_THRESHOLD, feature_type=feature_type, classifier_type=classifier_type, cropping_ratio=cropping_ratio)

        frameIdx = 0
        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            t2.run(img_resize)

            bboxListT2 = []
            for trackRes in t2.current_tracks:
                id = trackRes.id
                x1 = trackRes.tl[0]
                y1 = trackRes.tl[1]
                x2 = trackRes.br[0]
                y2 = trackRes.br[1]

                last_state = trackRes.last_state

                type = trackRes.type

                bbox = (x1, y1, x2, y2)
                bboxListT2.append((bbox, id, last_state, type))

            print('Processing ' + str(frameIdx))

            if (SAVEBBOX == True):
                outParentPath = out_parent_path
                if (not os.path.exists(outParentPath)):
                    os.mkdir(outParentPath)
                if (not os.path.exists(os.path.join(outParentPath, dataset_name))):
                    os.mkdir(os.path.join(outParentPath, dataset_name))
                saveBbox(os.path.join(outParentPath, dataset_name, eachFile.split('/').split('.')[0]), img, frameIdx,
                         bboxListT2)

            frameIdx += 1

def test_detection(parent_path, dataset_name, feature_type, classifier_type, out_parent_path='tmp',  color_bins=16):
    fileList = []

    def loadVideoPathFromParent():
        videoPathList = os.listdir(parent_path)

        for eachVideoPath in videoPathList:
            eachVideoAbsPath = os.path.join(parent_path, eachVideoPath)

            verified_output_path = os.path.join(out_parent_path, dataset_name, eachVideoPath.split('/')[-1].split('.')[0])

            #exclude the done file
            #if os.path.exists(verified_output_path):
            #    continue

            fileList.append(eachVideoAbsPath)

    loadVideoPathFromParent()



    for eachFile in fileList:
        print('Processing ' + eachFile)
        dp = DataProvider.VideoDataProvider(eachFile)

        resize_rate = 0.4

        t2 = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                      min_det_treshold=MIN_DET_THRESHOLD, use_cmatch=True, tracker_type='TM', tracker_limit=10,
                      feature_type=feature_type, classifier_type=classifier_type, cropping_ratio=cropping_ratio)

        frameIdx = 0
        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            t2.run(img_resize)

            bboxListT2 = []
            for trackRes in t2.current_tracks:
                id = trackRes.id
                x1 = trackRes.tl[0]
                y1 = trackRes.tl[1]
                x2 = trackRes.br[0]
                y2 = trackRes.br[1]

                last_state = trackRes.last_state

                type = trackRes.type

                bbox = (x1, y1, x2, y2)
                bboxListT2.append((bbox, id, last_state, type))

            print('Processing ' + str(frameIdx))

            if (SAVEBBOX == True):
                outParentPath = out_parent_path
                if (not os.path.exists(outParentPath)):
                    os.mkdir(outParentPath)
                if (not os.path.exists(os.path.join(outParentPath, dataset_name))):
                    os.mkdir(os.path.join(outParentPath, dataset_name))
                saveBbox(os.path.join(outParentPath, dataset_name, eachFile.split('/')[-1] + '_t2'), img, frameIdx,
                         bboxListT2)

            frameIdx += 1

def test_vino_reid():
    #parent_path = 'data/human_track_data_kaist_certification/video'
    parent_path = 'data/human_track_data_kaist_realenv/video'
    dataset_name = 'human_track_real_data_kaist_vino_reid'
    feature_type = 'vino_reid'
    classifier_type = 'KNC'
    out_parent_path = '210602_result'

    #test_c - 01 - 01 - 3p_matching

    test_detection(parent_path, dataset_name, feature_type, classifier_type, out_parent_path)

def tmp_vino_reid():
    # parent_path = 'data/human_track_data_kaist_certification/video'
    parent_path = 'testSample/video/'
    dataset_name = 'vino-reid-testSample'
    feature_type = 'vino_reid'
    classifier_type = 'KNC'
    out_parent_path = 'testSample/res'

    test_detection(parent_path, dataset_name, feature_type, classifier_type, out_parent_path)

def own_vino_reid():
    parent_path = 'data/human_track_data_kaist_realenv/video'
    #parent_path = 'testSample/video/'
    dataset_name = 'human_track_real_data_kaist_vino_reid_lhw'
    feature_type = 'vino_reid'
    classifier_type = 'KNC'
    out_parent_path = '210602_result'

    own_test_detection(parent_path, dataset_name, feature_type, classifier_type, out_parent_path)

def own_vino_demo_single(video_path, feature_type, classifier_type):

    dp = DataProvider.VideoDataProvider(video_path)
    resize_rate = 0.4

    t1 = TrackMod_own(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD,
                      min_det_treshold=MIN_DET_THRESHOLD, feature_type=feature_type, classifier_type=classifier_type,
                      cropping_ratio=cropping_ratio)

    frame_idx = 0

    while True:
        img = dp.get()

        if img is None:
            break
        img_resize = cv2.resize(img, (0,0), fx=resize_rate, fy=resize_rate)

        t1.run(img_resize)

        for track_res in t1.current_tracks:
            id = track_res.id
            x1 = int(track_res.tl[0])
            y1 = int(track_res.tl[1])
            x2 = int(track_res.br[0])
            y2 = int(track_res.br[1])



            cv2.rectangle(img_resize, (x1, y1), (x2, y2),(255,0,0), 3)
            cv2.putText(img_resize, str(id),(int((x1+x2)/2.0), y1+10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255))



        cv2.imshow('res', img_resize)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        if key == ord('c'):
            cv2.waitKey()





def own_vino_demo(video_path):
    feature_type = 'vino_reid'
    classifier_type = 'KNC'

    own_vino_demo_single(video_path, feature_type, classifier_type)


def main(argv=None):
    # VINO REID DEMO
    video_path = 'testSample/video/lab02-18-2p.mp4'
    own_vino_demo(video_path)



if __name__ == "__main__":
    sys.exit(main())


