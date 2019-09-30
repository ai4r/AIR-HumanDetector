import os
import cv2
import xml.etree.ElementTree as XMLTree


class VideoDataProvider:

    def __init__(self, path):
        self.filename = path
        self.frameNo = 0

        #if not os.path.exists(path):
        #    raise Exception('%s is not exist' % (path))

        v = cv2.VideoCapture()
        v.open(path)

        if not v.isOpened() :
            raise Exception('%s can not be opened' % (path))

        self.video = v
        self.totalFrameNo = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        if self.video.isOpened() :
            self.video.release()

    def get(self):
        ret, img = self.video.read()
        self.frameNo += 1
        return img

class ImageDataProvider:

    def __init__(self, path):
        self.path = path
        self.frameNo = 0
        self.filenames = []

        for fileName in os.listdir(path):
            img = cv2.imread(self.path + '/' + fileName)
            if img is None:
                continue
            else:
                self.filenames.append(fileName)
        self.filenames.sort()
        self.totalFrameNo = len(self.filenames)

    def get(self):
        if self.frameNo < len(self.filenames):
            self.img = cv2.imread(self.path + '/' + self.filenames[self.frameNo])
            self.frameNo += 1
            return self.img
        else:
            return None

class IllmenauDataProvider:

    def __init__(self, path, target):
        if target == 'Chair':
            img_dir = 'HomeLab-ChairImg'
        elif target == 'Couch':
            img_dir = 'HomeLab-CouchImg'
        elif target == 'Follow':
            img_dir = 'FollowImg'
        elif target == 'Hallway':
            img_dir = 'Floor-3PersonsImg'
        elif target == 'Sitting1':
            img_dir = 'HomeLab-Sitting1Img'
        elif target == 'Sitting2':
            img_dir = 'HomeLab-Sitting2Img'
        elif target == 'Sitting3':
            img_dir = 'HomeLab-Sitting3Img'
        elif target == 'Sitting4':
            img_dir = 'HomeLab-Sitting4Img'
        else:
            raise Exception('Invalid target = ' + target)

        self.image_path = path + '/images/' + img_dir
        for (dirpath, dirnames, filenames) in os.walk(self.image_path):
            filenames.sort()
            self.filename = filenames
            self.dirPath = dirpath
            self.frameNo = 0

        self.annotation_path = path + '/labels/' + target + '.xml'
        tree = XMLTree.parse(self.annotation_path)
        root = tree.getroot()

        class Track:
            pass

        self.tracks = []
        for track in root.findall('track'):
            track_list = []

            t_id = int(track.get('id'))
            for box in track.findall('box'):

                x1 = int(box.get('xtl'))
                x2 = int(box.get('xbr'))
                y1 = int(box.get('ytl'))
                y2 = int(box.get('ybr'))
                frameNo = int(box.get('frame'))
                is_outside = (box.get('outside') == '1')
                is_occluded = (box.get('occluded') == '1')

                t = Track()
                t.id = t_id
                t.name = 'person'
                t.frameNo = frameNo
                t.tl = (x1, y1)
                t.br = (x2, y2)
                t.is_outside = is_outside
                t.is_occluded = is_occluded
                t.is_marked = False
                track_list.append(t)

            self.tracks.append(track_list)

    def get(self):
        if self.frameNo < len(self.filename):
            img = cv2.imread(self.dirPath + '/' + self.filename[self.frameNo])
            self.current_tracks = []
            for track in self.tracks:
                self.current_tracks.append(track[self.frameNo])

            self.frameNo += 1
            return img
        return None

class INRIADataProvider:

    def __init__(self, path):
        for (dirpath, dirnames, filenames) in os.walk(path + "/Train/pos"):
            self.filename = filenames
            self.dirPath = dirpath
            self.frameNo = 0

    def get(self):
        if self.frameNo < len(self.filename):
            img = cv2.imread(self.dirPath + '/' + self.filename[self.frameNo])
            self.frameNo += 1
            return img
        return None

class VOCDataProvider:

    def __init__(self, path):
        self.path = path
        self.frameNo = 0
        self.annotation_files = os.listdir(path + '/Annotations')
        self.totalFrameNo = len(self.annotation_files)

    def get(self):
        if self.frameNo < len(self.annotation_files):
            annotation_path = self.path + '/Annotations/' + self.annotation_files[self.frameNo]
            tree = XMLTree.parse(annotation_path)
            root = tree.getroot()

            file_path = self.path + '/Images/' + root.find('filename').text
            img = cv2.imread(file_path)

            if img is None:
                raise Exception('Invalid Image - ' + str(file_path))

            class Track:
                pass

            self.current_tracks = []
            for obj in root.findall('object'):
                box = obj.find('bndbox')

                name = obj.find('name').text
                x1 = int(box.find('xmin').text)
                x2 = int(box.find('xmax').text)
                y1 = int(box.find('ymin').text)
                y2 = int(box.find('ymax').text)

                t = Track()
                t.frameNo = self.frameNo
                t.name = name
                t.tl = (x1, y1)
                t.br = (x2, y2)
                t.is_outside = False
                t.is_occluded = False
                t.is_marked = False

                self.current_tracks.append(t)

            self.frameNo += 1
            return img
        return None