import os
import cv2
import xml.etree.ElementTree as XMLTree

class VideoDataProvider:

    def __init__(self, path, resize_rate=1.0, start_frame=0, end_frame=1e+20):
        self.filePath = path
        self.frameNo = 10
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.resize_rate = resize_rate

        if type(path) == str:
            self.fileName = path.split('/')[-1].split('.')[0]
            if not os.path.exists(path):
                raise Exception('%s is not exist' % (path))

        v = cv2.VideoCapture()
        v.open(path)

        if not v.isOpened() :
            raise Exception('%s can not be opened' % str(path))

        self.video = v
        self.totalFrameNo = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

        if start_frame > 0:
            self.jump(start_frame)

    def __del__(self):
        if self.video.isOpened() :
            self.video.release()

    def get(self):
        if self.frameNo < self.end_frame:
            ret, img = self.video.read()
            if img is not None and self.resize_rate != 1.0:
                img = cv2.resize(img, (0, 0), fx=self.resize_rate, fy=self.resize_rate)
            self.frameNo += 1
            return img
        return None

    def jump(self, frame_pos):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.frameNo = frame_pos

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
        target_dirs = []
        target_names = []
        if target == 'Chair':
            target_names.append(target)
            target_dirs.append('HomeLab-ChairImg')
        elif target == 'Couch':
            target_names.append(target)
            target_dirs.append('HomeLab-CouchImg')
        elif target == 'Follow':
            target_names.append(target)
            target_dirs.append('FollowImg')
        elif target == 'Hallway':
            target_names.append(target)
            target_dirs.append('Floor-3PersonsImg')
        elif target == 'Sitting1':
            target_names.append(target)
            target_dirs.append('HomeLab-Sitting1Img')
        elif target == 'Sitting2':
            target_names.append(target)
            target_dirs.append('HomeLab-Sitting2Img')
        elif target == 'Sitting3':
            target_names.append(target)
            target_dirs.append('HomeLab-Sitting3Img')
        elif target == 'Sitting4':
            target_names.append(target)
            target_dirs.append('HomeLab-Sitting4Img')
        elif target == 'All':
            target_names = ['Chair', 'Couch', 'Follow', 'Hallway', 'Sitting1', 'Sitting2', 'Sitting3', 'Sitting4']
            target_dirs = ['HomeLab-ChairImg', 'HomeLab-CouchImg', 'FollowImg', 'Floor-3PersonsImg',
                           'HomeLab-Sitting1Img', 'HomeLab-Sitting2Img', 'HomeLab-Sitting3Img', 'HomeLab-Sitting4Img']
        else:
            raise Exception('Invalid target = ' + target)

        self.filename = []
        self.tracks = []
        self.frameNo = 0

        for idx, img_dir in enumerate(target_dirs):
            target = target_names[idx]

            cur_files = []
            self.image_path = path + '/images/' + img_dir
            for (dirpath, dirnames, filenames) in os.walk(self.image_path):
                filenames.sort()
                for f_name in filenames:
                    cur_files.append(dirpath + '/' + f_name)

            annotation_path = path + '/labels/' + target + '.xml'
            tree = XMLTree.parse(annotation_path)
            root = tree.getroot()

            class Track:
                pass

            cur_tracks = []
            for i in range(len(cur_files)):
                cur_tracks.append([])

            for track in root.findall('track'):
                t_id = int(track.get('id'))
                for box in track.findall('box'):
                    frame = int(box.get('frame'))
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
                    cur_tracks[frame].append(t)

            self.filename.extend(cur_files)
            self.tracks.extend(cur_tracks)

        self.current_tracks = []

    def get(self):
        if self.frameNo < len(self.filename):
            img = cv2.imread(self.filename[self.frameNo])

            self.current_tracks = []
            for t in self.tracks[self.frameNo]:
                self.current_tracks.append(t)

            self.frameNo += 1
            return img
        return None


class LivingLabDataProvider:
    def __init__(self, path, no, resize_rate=0.5):
        self.dirPath = path
        self.data_no = no
        self.frameNo = 0
        self.resize_rate = resize_rate

        class Track:
            pass

        self.filename = []
        self.tracks = []

        for (dirpath, dirnames, filenames) in os.walk('%s/%d/Images' % (path, no)):
            for idx, file in enumerate(filenames):
                self.filename.append(file)
                annotation_path = '%s/%d/Annotations/%s.xml' % (path, no, file[:-4])
                tree = XMLTree.parse(annotation_path)
                root = tree.getroot()

                # img = cv2.imread('%s/%d/Images/%s' % (self.dirPath, self.data_no, file))
                img_width = int(root.find('size').find('width').text)
                img_height = int(root.find('size').find('height').text)

                pid = 0
                track_list = []
                for track in root.findall('object'):
                    t_id = pid
                    for box in track.findall('bndbox'):
                        x1 = int(box.find('xmin').text) * resize_rate
                        y1 = int(box.find('ymin').text) * resize_rate
                        x2 = int(box.find('xmax').text) * resize_rate
                        y2 = int(box.find('ymax').text) * resize_rate
                        is_outside = False
                        is_occluded = False

                        # x2 = x1 + x2
                        # y2 = y1 + y2
                        # box.find('xmax').text = str(x2)
                        # box.find('ymax').text = str(y2)

                        if x1 <= 0 or x2 >= img_width or y1 <= 0 or y2 >= img_height:
                            continue

                        t = Track()
                        t.id = t_id
                        t.name = 'person'
                        t.frameNo = idx
                        t.tl = (int(x1), int(y1))
                        t.br = (int(x2), int(y2))
                        t.is_outside = is_outside
                        t.is_occluded = is_occluded
                        t.is_marked = False
                        track_list.append(t)

                        # cv2.rectangle(img, t.tl, t.br, (255, 255, 255), 1)

                    pid += 1

                # cv2.imshow('Image', img)
                # cv2.waitKey(10)

                # import Utilities as util
                # util.indentXML(root)
                # XMLTree.ElementTree(root).write(annotation_path)

                self.tracks.append(track_list)

    def get(self):
        if self.frameNo < len(self.filename):
            img = cv2.imread('%s/%d/Images/%s' % (self.dirPath, self.data_no, self.filename[self.frameNo]))

            if self.resize_rate != 1.0:
                img = cv2.resize(img, (0,0), fx=self.resize_rate, fy=self.resize_rate)

            self.current_tracks = []
            for track in self.tracks[self.frameNo]:
                self.current_tracks.append(track)
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

