import os
import time
import sys
import cv2
import DataProvider
import numpy as np

from calcHistogram import calcHistogram3D
from calcHistogram import calcOrigin



import tracking.Tracking_old as Tracking_old
import tracking.Tracking as Tracking

from darkflow.net.build import TFNet


event_log = []
online_pos_images = []
online_neg_images = []

#SAVEBBOX = True

SAVEBBOX = False

cropping_ratio = (0.8, 0.8)

DET_THRESHOLD=0.2
MIN_DET_THRESHOLD=0.1

origin_resize_ratio = (32, 32)


FEATURETYPE = 'colorHist'
#FEATURETYPE = 'yolo'
#FEATURETYPE = 'origin'

#CLASSIFIERTYPE = 'SVC'
CLASSIFIERTYPE = 'KNC'



# feature type: yolo, colorHist

colorBins = 0

class TrackMod_dual:
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD,tracker_type='',
                 tracker_limit=60, use_cmatch=False):
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
                                                            tracker_limit=tracker_limit, featureType=FEATURETYPE,
                                                            classifierType=CLASSIFIERTYPE)
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
    def get_feature_color(self,img, rectList):
        f = calcHistogram3D(img, rectList, cropping_ratio)
        return f
    def get_feature_origin(self, img, rectList):
        f = calcOrigin(img, rectList, resize_ratio=origin_resize_ratio)
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

        while len(online_neg_images) > 17:
            online_neg_images.remove(online_neg_images[0])

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

        while len(online_neg_images) > 17:
            online_neg_images.remove(online_neg_images[0])

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


class TrackMod:
    def __init__(self, conf_file, model_file, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD, tracker_type='',
                 tracker_limit=60, use_cmatch=False, old_tracker = False):

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
        if old_tracker:
            self.featureType = 'yolo'
            self.tracking = Tracking_old.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                              tracker_limit=tracker_limit, featureType=self.featureType)
        else:
            self.featureType = FEATURETYPE
            self.tracking = Tracking.Tracking(tracker_type=tracker_type, use_cmatch=use_cmatch,
                                              tracker_limit=tracker_limit, featureType=self.featureType,
                                              classifierType=CLASSIFIERTYPE)



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
            f = calcHistogram3D(img, rectList, cropping_ratio)

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

            cv2.putText(img, 'ID: %d' % tr.id, (tr.tl[0] + 20, tr.tl[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.rectangle(img, tr.tl, tr.br, c, 4)

    def close(self):
        self.det_net = None

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

def tmp(argv=None):
    file_list = ['./testSample/video/lab02-18-2p.mp4']

    for each_file in file_list:
        dp = DataProvider.VideoDataProvider(each_file)


        resize_rate = 0.4
        t2 = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000, det_threshold=DET_THRESHOLD, min_det_treshold=MIN_DET_THRESHOLD,
                      use_cmatch=True, tracker_type='TM', tracker_limit=10)

        t2.tracking.event_func = event_handler
        t2.show_online_data = True

        frameIdx = 0
        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            # t1.run(img_resize)
            t2.run(img_resize)

            bboxListT1 = []
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
                if (not os.path.exists('testSample/out')):
                    os.mkdir('testSample/out')
                saveBbox('testSample/out', img, frameIdx, bboxListT2)

            frameIdx += 1


def main(argv=None):
    #colorBins = 8



    colorBins = 16
    #colorBins = 32

    parentPath = '../data/human_track_data_kaist/video'
    datasetName = 'human_track_data_kaist-color_' + str(colorBins)



    fileList = []

    def loadVideoPathFromParent():
        videoPathList = os.listdir(parentPath)

        for eachVideoPath in videoPathList:
            eachVideoAbsPath = os.path.join(parentPath, eachVideoPath)

            fileList.append(eachVideoAbsPath)

    loadVideoPathFromParent()



    '''
    fileList = ['../data/problem/Tracker(2crosssometimes).mp4', '../data/problem/Tracker(2atthesametime).mp4',
                '../data/problem/Tracker(2crossoften).mp4', '../data/normal/20181128_134433.AVI',
                '../data/normal/20181128_134647.AVI', '../data/normal/20181128_134959.AVI']
    '''

    #fileList = ['../data/problem/Tracker(2crosssometimes).mp4']

    for eachFile in fileList:
        dp = DataProvider.VideoDataProvider(eachFile)

        resize_rate = 0.4

        '''
        t1 = TrackMod(conf_file="cfg/yolo.cfg", model_file='bin/yolo.weights',
                      det_threshold=0.5, min_det_treshold=0.5,
                      use_cmatch=False, tracker_type='TM', tracker_limit=60)
        '''

        t2 = TrackMod(conf_file="cfg/yolo-f.cfg", model_file=8000,det_threshold=0.2, min_det_treshold=0.1,use_cmatch=True, tracker_type='TM', tracker_limit=10)

        t2.tracking.event_func = event_handler
        t2.show_online_data = True

        frameIdx= 0
        while True:
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resize_rate, fy=resize_rate)

            #t1.run(img_resize)
            t2.run(img_resize)

            bboxListT1 = []
            '''
            for trackRes in t1.current_tracks:
                id = trackRes.id
                x1 = trackRes.tl[0]
                y1 = trackRes.tl[1]
                x2 = trackRes.br[0]
                y2 = trackRes.br[1]



                last_state = trackRes.last_state

                type = trackRes.type

                bbox = (x1,y1,x2,y2)
                bboxListT1.append((bbox, id, last_state, type))
            #saveBbox(eachFile.split('/')[-1] + '_t1',img, frameIdx, bboxListT1)
            '''

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

            if(SAVEBBOX == True):
                outParentPath = 'out_201030'
                if(not os.path.exists(outParentPath)):
                    os.mkdir(outParentPath)
                if(not os.path.exists(os.path.join(outParentPath, datasetName))):
                    os.mkdir(os.path.join(outParentPath, datasetName))
                saveBbox(os.path.join(outParentPath, datasetName, eachFile.split('/')[-1] + '_t2'),img, frameIdx, bboxListT2)


            frameIdx += 1



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    #sys.exit(main())
    sys.exit(tmp())

