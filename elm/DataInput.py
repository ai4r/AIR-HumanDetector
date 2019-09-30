# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import csv
import random
import pickle

import tensorflow as tf
import numpy as np

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin

import cv2
import elm.Utilities as util


def maybe_download(src_url, filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(src_url + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels

class DataRecord(object):
    def __init__(self, data=None, label=None):
        self.data = data
        self.label = label

class Option(object):
    pass

class DistortionOption(Option):
    def __init__(self):
        self.do_shuffle = True
        self.distortion_per_image = 100
        self.distortion_pool_size = 10000
        self.rotation = []
        self.crop = []
        self.do_standardization = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.brightness_delta = 0.0
        self.contrast_range = []
        self.images = None

class DataSets(object):
    def resize(self, w, h):
        if hasattr(self, 'train'):
            self.train.resize(w, h)
        if hasattr(self, 'validation'):
            self.validation.resize(w, h)
        if hasattr(self, 'test'):
            self.test.resize(w, h)

    @staticmethod
    def create(train_data=None, train_label=None, test_data=None, test_label=None, to_row_vectors=False):
        ds = DataSets()

        if train_data is not None:
            ds.train = DataSet(train_data, train_label, to_row_vectors=to_row_vectors)
        if test_data is not None:
            ds.test = DataSet(test_data, test_label, to_row_vectors=to_row_vectors)

        return ds

    @classmethod
    def distort_images(self, data, distortion_per_image, target_image_num, rot=[], crop=[],
                       do_standardization=False,
                       flip_horizontal=False, flip_vertical=False,
                       brightness_delta=0.0, contrast_range=[], hue_delta=0,
                       dump_file='', dump_division=10000,
                       include_origianl=False, do_shuffle=False, display_step=0) :

        num_of_images = data.num_examples
        distorted_images = []
        labels = []
        div_num = 0

        min_w = min_h = max_w = max_h = 0
        keep_ratio = False
        if len(crop) > 1 :
            min_w = int(data.width * crop[0])
            max_w = int(data.width * crop[1])
            keep_ratio = True

        if len(crop) > 2:
            min_h = int(data.width * crop[2])
            max_h = int(data.width * crop[3])

        min_r = max_r = 0
        if len(rot) > 0:
            min_r = rot[0]
            max_r = rot[1]

        current_count = 0
        util.record_time()

        if include_origianl:
            target_image_num += data.num_examples

        g = tf.Graph()
        with g.as_default():
            with tf.Session() as sess:
                coord = tf.train.Coordinator()
                thread = None

                v_idx = tf.Variable(0)
                sess.run(tf.global_variables_initializer())

                def data_augmentation_task():
                    distorted_image = data.images[idx]
                    label = data.labels[idx]

                    # Random Rotation
                    if max_r > 0:
                        r = random.randrange(min_r, max_r + 1)
                        distorted_image = tf.contrib.image.rotate(distorted_image, r * (180.0 / 3.1415926535))
                        # distorted_image = ndimage.rotate(distorted_image, r, reshape=False)

                    # Random Crop
                    if min_w > 0:
                        w = random.randrange(min_w, max_w + 1)
                        h = random.randrange(min_h, max_h + 1)

                        if keep_ratio:
                            h = w

                        distorted_image = tf.random_crop(distorted_image, [h, w, data.channel])
                        distorted_image = tf.reshape(distorted_image, [1, h, w, data.channel])
                        distorted_image = tf.image.resize_bilinear(distorted_image, size=(data.height, data.width))
                        distorted_image = tf.reshape(distorted_image, [data.height, data.width, data.channel])

                    # Randomly flip the image horizontally.
                    if flip_horizontal:
                        distorted_image = tf.image.random_flip_left_right(distorted_image)

                    if flip_vertical:
                        distorted_image = tf.image.random_flip_up_down(distorted_image)

                    # Because these operations are not commutative, consider randomizing
                    # the order their operation.
                    # NOTE: since per_image_standardization zeros the mean and makes
                    # the stddev unit, this likely has no effect see tensorflow#1458.
                    if brightness_delta != 0:
                        distorted_image = tf.image.random_brightness(distorted_image, max_delta=brightness_delta)

                    if len(contrast_range) > 0:
                        distorted_image = tf.image.random_contrast(distorted_image, lower=contrast_range[0],
                                                                   upper=contrast_range[1])

                    if hue_delta > 0:
                        distorted_image = tf.image.random_hue(distorted_image, hue_delta)

                    # Subtract off the mean and divide by the variance of the pixels.
                    if do_standardization:
                        distorted_image = tf.image.per_image_standardization(distorted_image)

                    task_batch = tf.train.batch([distorted_image, label], distortion_per_image,
                                                capacity=distortion_per_image * 2, num_threads=32)
                    return task_batch

                while current_count < target_image_num:

                    if include_origianl == True and current_count < data.num_examples:
                        distorted_images.append(data.images[current_count])
                        labels.append(data.labels[current_count])
                        current_count += 1

                    else:

                        r_idx = np.random.randint(0, num_of_images, dtype=np.int32)
                        task_op = data_augmentation_task(r_idx)

                        tf.train.start_queue_runners(sess=sess, coord=coord)
                        b_imgs, b_labels = sess.run(task_op)

                        distorted_images.extend(b_imgs)
                        labels.extend(b_labels)
                        current_count += len(b_imgs)

                    if display_step > 0 and current_count % display_step == 0 :
                        print('\t%d processed (%s)' % (current_count, util.get_elapsed_time()))

                    if dump_file != '' and len(distorted_images) >= dump_division:
                        if do_shuffle:
                            DataSets.shuffle(distorted_images, labels)

                        DataSets.dump_file(div_num, dump_file, distorted_images, labels,
                                           data.dim_features, data.dim_labels, data.width, data.height, data.channel,
                                           target_image_num)
                        div_num += 1
                        distorted_images.clear()
                        labels.clear()

                coord.request_stop()
                coord.join(thread)

        if do_shuffle:
            DataSets.shuffle(distorted_images, labels)

        if dump_file == '':
            distorted_images = np.array(distorted_images)
            labels = np.array(labels)
            return distorted_images, labels

        elif len(distorted_images) > 0:
            DataSets.dump_file(div_num, dump_file, distorted_images, labels,
                               data.dim_features, data.dim_labels, data.width, data.height, data.channel,
                               target_image_num)

    @classmethod
    def dump_file(cls, dump_no, file_pattern, features, labels, dim_features, dim_labels, width, height, channel,
                  total_count=0):
        dict = {}

        if dump_no == 0:
            dict['total_count'] = total_count

        dict['no'] = dump_no
        dict['width'] = width
        dict['height'] = height
        dict['channel'] = channel

        dict['count'] = len(features)
        dict['dim_features'] = dim_features
        dict['dim_labels'] = dim_labels
        dict['data'] = np.array(features)
        dict['labels'] = np.array(labels)

        filename = file_pattern % dump_no
        pickle.dump(dict, open(filename, 'wb'))

    @classmethod
    def dump(cls, features, labels, file_pattern, division_count = -1, width=-1, height=1, channel=1):
        total_count = 0

        if width > 0 :
            w = width
        else:
            w = features[0].shape[1]
        h = height
        c = channel

        if len(features[0].shape) > 2:
            h = features[0].shape[2]

        if len(features[0].shape) > 3:
            c = features[0].shape[3]

        dim_features = w * h * c
        dim_labels = labels[0].shape[1]

        if len(features) != len(labels) :
            raise Exception('num of features != num of labels')

        for i in range(len(features)):
            total_count += len(features[i])

        cur_count = 0
        div_count = 0
        cur_item_idx = 0
        cur_list_idx = 0

        div_f = []
        div_l = []


        while cur_count < total_count:
            div_f.append(features[cur_list_idx][cur_item_idx])
            div_l.append(labels[cur_list_idx][cur_item_idx])

            cur_count += 1
            cur_item_idx += 1

            if division_count > 0 and len(div_f) >= division_count:
                DataSets.dump_file(division_count, file_pattern, div_f, div_l, dim_features, dim_labels,
                                   w, h, c, total_count)
                div_f.clear()
                div_l.clear()
                div_count += 1

            if cur_item_idx >= len(features[cur_list_idx]):
                cur_item_idx = 0
                cur_item_idx += 1

        if len(div_f) > 0:
            DataSets.dump_file(division_count, file_pattern, div_f, div_l, dim_features, dim_labels,
                               w, h, c, total_count)

    @classmethod
    def shuffle(self, a, b):
        perm = np.arange(len(a))
        np.random.shuffle(perm)

        for i in range(int(len(a) / 2)):
            idx1 = perm[i*2]
            idx2 = perm[(i*2)+1]

            temp = a[idx1].copy()
            a[idx1] = a[idx2]
            a[idx2] = temp

            temp = b[idx1].copy()
            b[idx1] = b[idx2]
            b[idx2] = temp


    @classmethod
    def load(cls, file_pattern):
        d = DumpedDataSet(file_pattern)
        return d

    def clone(self):
        cl = DataSets()
        if hasattr(self, 'train'):
            cl.train = self.train.clone()
        if hasattr(self, 'validation'):
            cl.train = self.validation.clone()
        if hasattr(self, 'test'):
            cl.test = self.test.clone()
        return cl


class DataSet(object):
    def __init__(self, images, labels, to_row_vectors=True):
        if labels is not None:
            assert images.shape[0] == labels.shape[0], \
                ('# Of Samples(=%d) != # of Labels(=%d)' % (images.shape[0], labels.shape[0]))

        self.labels = None
        self.images = None
        self.depth = 0

        self.set_images(images, to_row_vectors=to_row_vectors)
        if labels is None:
            self.set_labels(images)
        else:
            self.set_labels(labels)

        self.original_index = np.arange(0, self.images.shape[0])
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._batch_index = -1
        self.session = None

        self.option = Option()
        self.option.batch_size = 0
        self.data_augmenter = None
        self.original_images = None
        self.original_labels = None
        self.normalization = None

    def set_images(self, images, to_row_vectors=False):
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        img_shape = images.shape

        is_row_vector = len(img_shape) == 2
        if to_row_vectors:
            # (samples x height x width x channel) -> (samples x feature_dim)
            if not is_row_vector:
                if len(img_shape) == 3:
                    images = images.reshape(img_shape[0], img_shape[1] * img_shape[2])
                    self.channel = 1
                elif len(img_shape) == 4:
                    images = images.reshape(img_shape[0], img_shape[1] * img_shape[2] * img_shape[3])
                    self.channel = img_shape[3]
                else:
                    raise Exception('Only 1 channel and 3 channel are supported')

                self.width = img_shape[2]
                self.height = img_shape[1]

            # (samples x feature_dim)
            else:
                self.width = img_shape[1]
                self.height = 1
                self.channel = 1
            self.is_image = False

        else:
            # (samples x feature_dim)
            if is_row_vector:
                self.width = img_shape[1]
                self.height = 1
                self.channel = 1
                self.is_image = False

            # (samples x height x width x channel)
            else:
                self.width = img_shape[2]
                self.height = img_shape[1]
                if(len(img_shape) > 3):
                    self.channel = img_shape[3]
                else:
                    self.channel = 1
                self.is_image = True

        self.images = images

    def set_augmentation(self, aug_seq):
        self.augmentation = aug_seq
        self.aug_task = None

    def set_labels(self, labels):
        img_shape = labels.shape
        is_row_vector = len(img_shape) <= 2

        # (samples x height x width x channel) -> (samples x feature_dim)
        if not is_row_vector:
            if len(img_shape) == 2:
                labels = labels.reshape(img_shape[0], img_shape[1])
            elif len(img_shape) == 3:
                labels = labels.reshape(img_shape[0], img_shape[1] * img_shape[2])
            elif len(img_shape) == 4:
                labels = labels.reshape(img_shape[0], img_shape[1] * img_shape[2] * img_shape[3])
            else:
                raise Exception('Only 1 channel and 3 channel are supported')

        self.labels = labels

    @property
    def num_examples(self):
        s = self.images.shape
        return s[0]

    @property
    def dim_features(self):
        s = self.images.shape
        if len(s) == 2:
            return s[1]
        elif len(s) == 3:
            return s[1] * s[2]
        elif len(s) == 4:
            return s[1] * s[2] * s[3]

    @property
    def shape_features(self):
        s = self.images.shape
        if len(s) == 2:
            return s[1]
        elif len(s) == 3:
            return s[1], s[2]
        elif len(s) == 4:
            return s[1], s[2], s[3]

    @property
    def dim_labels(self):
        label_shape = np.shape(self.labels)
        if len(label_shape) > 1:
            return label_shape[1]
        else:
            return 1

    @property
    def shape_labels(self):
        s = self.labels.shape
        if len(s) == 2:
            return s[1]
        elif len(s) == 3:
            return s[1], s[2]
        elif len(s) == 4:
            return s[1], s[2], s[3]

    @property
    def batch_index(self):
        return self._batch_index

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def clone(self):
        cl = DataSet(self.images[:], self.labels[:], to_row_vectors=False)
        cl.original_index = self.original_index[:]
        cl.width = self.width
        cl.height = self.height
        cl.channel = self.channel
        return cl

    def shuffle(self):
        DataSet.shuffle_arrays([self.images, self.labels])

    @staticmethod
    def shuffle_arrays(list_of_array):
        num_examples = len(list_of_array[0])
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        for i in range(int(num_examples / 2)):
            idx1 = perm[i * 2]
            idx2 = perm[(i * 2) + 1]

            for j in range(len(list_of_array)):
                if type(list_of_array[j]) is np.ndarray:
                    temp = list_of_array[j][idx1].copy()
                else:
                    temp = list_of_array[j][idx1]
                list_of_array[j][idx1] = list_of_array[j][idx2]
                list_of_array[j][idx2] = temp

    def truncate(self, num):
        if num < self.num_examples :
            self.images = self.images[:num]
            self.labels = self.labels[:num]

    def resize(self, w, h):
        images = self.images.reshape(self.num_examples, self.width, self.height, self.channel)
        resize_op = tf.image.resize_bilinear(images, [w, h])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            images = sess.run(resize_op)

        images = images.reshape(self.num_examples, w * h * self.channel)
        self.images = images
        self.width = w
        self.height = h

    def convert_color_space(self, from_space, to_space):
        if from_space == 'RGB' and to_space == 'LAB':
            conv_type = cv2.COLOR_RGB2LAB
        elif from_space == 'RGB' and to_space == 'GRAY':
            conv_type = cv2.COLOR_RGB2GRAY
        elif from_space == 'BGR' and to_space == 'LAB':
            conv_type = cv2.COLOR_BGR2LAB
        elif from_space == 'BGR' and to_space == 'GRAY':
            conv_type = cv2.COLOR_BGR2GRAY
        elif from_space == 'RGB' and to_space == 'HSV':
            conv_type = cv2.COLOR_RGB2HSV
        elif from_space == 'BGR' and to_space == 'HSV':
            conv_type = cv2.COLOR_BGR2HSV
        elif from_space == 'RGB' and to_space == 'YUV':
            conv_type = cv2.COLOR_RGB2YUV
        elif from_space == 'BGR' and to_space == 'YUV':
            conv_type = cv2.COLOR_BGR2YUV
        elif from_space == 'LAB' and to_space == 'RGB':
            conv_type = cv2.COLOR_LAB2RGB
        elif from_space == 'LAB' and to_space == 'BGR':
            conv_type = cv2.COLOR_LAB2BGR
        elif from_space == 'HSV' and to_space == 'RGB':
            conv_type = cv2.COLOR_HSV2RGB
        elif from_space == 'HSV' and to_space == 'BGR':
            conv_type = cv2.COLOR_HSV2BGR
        elif from_space == 'YUV' and to_space == 'RGB':
            conv_type = cv2.COLOR_YUV2RGB
        elif from_space == 'YUV' and to_space == 'BGR':
            conv_type = cv2.COLOR_YUV2BGR
        else:
            raise Exception('Not Supported conversion!')

        for i in range(self.num_examples):
            cv2.cvtColor(self.images[i], conv_type, self.images[i])

    def shapeToImage(self):
        if not self.is_image:
            self.images = np.array(util.vectors_to_image(self.images, self.width, self.height, self.channel, do_normalize=False))
            self.is_image = True

    def shapeToVector(self):
        if self.is_image:
            self.images = self.images.reshape(self.num_examples, self.width * self.height * self.channel)
            self.is_image = False

    def sort(self):
        label_idx = np.argmax(self.labels, axis=1)
        sorted_idx = np.argsort(label_idx)
        self.images = self.images[sorted_idx]
        self.labels = self.labels[sorted_idx]

    def init(self):
        self.rewind()
        if self.original_images is not None:
            self.images = self.original_images
            self.labels = self.original_labels
            self.original_images = None
            self.original_labels = None

    def rewind(self):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._batch_index = -1

    def has_next_batch(self):
        return self._index_in_epoch < self.num_examples

    def close(self):
        del self.images
        del self.labels
        if self.original_images is not None:
            del self.original_images

    def next_batch(self, batch_size, do_shuffle=True):
        return self.normal_batch(batch_size, do_shuffle)

    def normal_batch(self, batch_size, do_shuffle=True):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self.num_examples:
            self._epochs_completed += 1

            # Shuffle the data
            if do_shuffle:
                self.shuffle()

                # Start form the begin
                start = 0
                self._index_in_epoch = batch_size
            else:
                # Squeeze last data
                if start < self.num_examples:
                    self._index_in_epoch = start + (self.num_examples - start)

                # Start form the begin
                else:
                    start = 0
                    self._index_in_epoch = batch_size

        end = self._index_in_epoch
        self._batch_index += 1
        return self.images[start:end], self.labels[start:end]


class DumpedDataSet(DataSet):

    def __init__(self, file_pattern, file_idx=0):
        DataSet.__init___(None, None)
        self.list_idx = file_idx
        self.file_list = []
        self.file_pattern = file_pattern

        for i in range(100000):
            filePath = (file_pattern % i)
            if os.path.isfile(filePath):
                self.file_list.append(filePath)
            else:
                break

        if len(self.file_list) == 0 :
            raise Exception('DumptedDataSet - no dump files!')

        self.load_file(0)
        self.original_index = np.arange(0, self.images.shape[0])
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def load_file(self, idx):
        dict = pickle.load(open(self.file_list[0], 'rb'))

        if idx == 0:
            self.total_count = dict['total_count']
            self.width = dict['width']
            self.height = dict['height']
            self.channel = dict['channel']
            self.d_features = dict['dim_features']
            self.d_labels = dict['dim_labels']

        self.images = dict['data']
        self.labels = dict['labels']
        self.original_index = np.arange(0, self.images.shape[0])
        self.is_image = len(self.images.shape) > 2

    @property
    def num_examples(self):
        return self.total_count

    @property
    def dim_features(self):
        return self.d_features

    @property
    def dim_labels(self):
        return self.d_labels

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def clone(self):
        raise Exception('DumpedDataset is not clonable!')

    def shuffle(self):
        current_sample_num = self.images.shape[0]
        perm = np.arange(current_sample_num)
        np.random.shuffle(perm)

        for i in range(int(current_sample_num / 2)):
            idx1 = perm[i*2]
            idx2 = perm[(i*2)+1]

            temp = self.images[idx1].copy()
            self.images[idx1] = self.images[idx2]
            self.images[idx2] = temp

            temp = self.labels[idx1].copy()
            self.labels[idx1] = self.labels[idx2]
            self.labels[idx2] = temp

            temp = self.original_index[idx1].copy()
            self.original_index[idx1] = self.original_index[idx2]
            self.original_index[idx2] = temp


    def resize(self, w, h):
        images = self.images.reshape(self.num_examples, self.width, self.height, self.channel)
        resize_op = tf.image.resize_bilinear(images, [w, h])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            images = sess.run(resize_op)

        images = images.reshape(self.num_examples, w * h * self.channel)
        self.images = images
        self.width = w
        self.height = h

    def shapeToImage(self):
        if not self.is_image:
            self.images = np.array(util.vectors_to_image(self.images, self.width, self.height, self.channel, do_normalize=False))
            self.is_image = True

    def shapeToVector(self):
        if self.is_image:
            self.images = self.images.reshape(self.num_examples, self.width * self.height)
            self.is_image = False

    def sort(self):
        label_idx = np.argmax(self.labels, axis=1)
        sorted_idx = np.argsort(label_idx)
        self.images = self.images[sorted_idx]
        self.labels = self.labels[sorted_idx]

    def next_batch(self, batch_size, do_shuffle=True):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > len(self.images):

            # Load next
            if self.list_idx < len(self.file_list) :
                self.list_idx += 1
                self.load_file(self.list_idx)

            # Start from the begin
            else:
                self.list_idx = 0
                self._epochs_completed += 1
                self.load_file(0)

            if do_shuffle:
                self.shuffle()

            if self.is_image:
                self.shapeToImage()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]

class StreamDataSet(DataSet):

    def __init__(self, filenames, width, height, channel, dim_labels, label_bytes,
                 num_examples, num_examples_per_epoch, one_hot=True, distortion_option=None):

        self.filenames = filenames

        self.width = width
        self.height = height
        self.channel = channel
        self.dim_of_labels = dim_labels
        self.num_of_examples = num_examples
        self.num_examples_per_epoch = num_examples_per_epoch
        self.label_bytes = label_bytes

        self.one_hot = one_hot
        self.distortion = distortion_option
        self.session = None
        self.coordinator = None
        self.graph = None
        self.thread = None
        self.batch_count = 0

    def inputs(self, batch_size, do_shuffle):
        for f in self.filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(self.filenames)

        # Read examples from files in the filename queue.
        read_input = self.read_data(filename_queue)
        reshaped_image = tf.cast(read_input.data, tf.float32)

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, self.height, self.width)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([self.height, self.width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return StreamDataSet._generate_image_and_label_batch(float_image, read_input.label,
                                                             min_queue_examples, batch_size,
                                                             shuffle=do_shuffle)

    def distorted_inputs(self, batch_size, do_shuffle):
        for f in self.filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(self.filenames)

        # Read examples from files in the filename queue.
        read_input = self.read_data(filename_queue)
        reshaped_image = tf.cast(read_input.data, tf.float32)

        height = self.height
        width = self.width
        channel = self.channel

        min_w = min_h = max_w = max_h = 0
        keep_ratio = False

        crop = self.distortion.crop
        rot = self.distortion.rotation

        if len(crop) > 1:
            min_w = int(width * crop[0])
            max_w = int(width * crop[1])
            keep_ratio = True

        if len(crop) > 2:
            min_h = int(height * crop[2])
            max_h = int(height * crop[3])

        min_r = max_r = 0
        if len(rot) > 0:
            min_r = rot[0]
            max_r = rot[1]

        # Image processing for training the network. Note the many random
        # distortions applied to the image.
        distorted_image = reshaped_image

        # Random Rotation
        if max_r > 0:
            r = random.randrange(min_r, max_r + 1)
            distorted_image = tf.contrib.image.rotate(distorted_image, r * (180.0 / 3.1415926535))

        # Randomly crop a [height, width] section of the image.
        if min_w > 0:
            w = random.randrange(min_w, max_w + 1)
            h = random.randrange(min_h, max_h + 1)
            if keep_ratio :
                h =  w

            distorted_image = tf.random_crop(distorted_image, [h, w, channel])
            distorted_image = tf.reshape(distorted_image, [1, h, w, channel])
            distorted_image = tf.image.resize_bilinear(distorted_image, size=(height, width))
            distorted_image = tf.reshape(distorted_image, [height, width, channel])

        if self.distortion.flip_horizontal:
            distorted_image = tf.image.random_flip_left_right(distorted_image)

        if self.distortion.flip_vertical:
            distorted_image = tf.image.random_flip_up_down(distorted_image)

        if self.distortion.brightness_delta != 0:
            distorted_image = tf.image.random_brightness(distorted_image, max_delta=self.distortion.brightness_delta)

        if self.distortion.contrast_range:
            distorted_image = tf.image.random_contrast(distorted_image, lower=self.distortion.contrast_range[0], upper=self.distortion.contrast_range[1])

        if self.distortion.do_standardization:
            distorted_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        distorted_image.set_shape([height, width, channel])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(self.num_examples_per_epoch * min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of examples.
        return StreamDataSet._generate_image_and_label_batch(distorted_image, read_input.label,
                                                             min_queue_examples, batch_size,
                                                             shuffle=do_shuffle)

    def read_data(self, filename_queue):
        result = DataRecord()

        label_bytes = self.label_bytes
        result.height = self.height
        result.width = self.width
        result.depth = self.channel
        image_bytes = result.height * result.width * result.depth

        # Every record consists of a label followed by the image, with a fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from uint8->int32.
        result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we reshape
        # from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.strided_slice(record_bytes, [label_bytes],
                             [label_bytes + image_bytes]),
            [result.depth, result.height, result.width])

        # Convert from [depth, height, width] to [height, width, depth].
        result.data = tf.transpose(depth_major, [1, 2, 0])
        return result

    @classmethod
    def _generate_image_and_label_batch(cls, image, label, min_queue_examples, batch_size, shuffle):
        num_preprocess_threads = 16

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size,
                min_after_dequeue=min_queue_examples)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)
        return images, tf.reshape(label_batch, [batch_size])

    def next_batch(self, batch_size, do_shuffle=True):
        if self.batch_count == 0:
            self.session = tf.Session()
            self.graph = tf.Graph()
            self.coordinator = tf.train.Coordinator()

            if self.distortion is None:
                self.batch_op = self.inputs(batch_size, do_shuffle)
            else:
                self.batch_op = self.distorted_inputs(batch_size, do_shuffle)

        self.batch_count += 1

        self.thread = tf.train.start_queue_runners(sess=self.session, coord=self.coordinator)
        images, labels = self.session.run(self.batch_op)

        if self.one_hot:
            labels = dense_to_one_hot(labels, self.dim_labels)

        return images, labels

    def init(self):
        self.batch_count = 0
        self.session = None

    def close(self):
        if self.session is not None:
            self.session.close()

        self.graph = None

        if self.coordinator is not None:
            self.coordinator.request_stop()
            self.coordinator.join(self.thread)
            self.coordinator = None

    def shapeToImage(self):
        pass

    def shapeToVector(self):
        pass

    @property
    def num_examples(self):
        return self.num_of_examples

    @property
    def dim_features(self):
        return self.width * self.height * self.channel

    @property
    def dim_labels(self):
        return self.dim_of_labels


def normalize(dataset, method="Avg-Var"):
    if method == "Avg-Var":
        mean = dataset.train.images.mean(0)
        std = dataset.train.images.std(0) + 1e-20
        dataset.train.images = (dataset.train.images - mean) / std

        if hasattr(dataset, 'test'):
            dataset.test.images = (dataset.test.images - mean) / std
        if hasattr(dataset, 'validation'):
            dataset.validation.images = (dataset.validation.images - mean) / std

    elif method == "Avg":
        mean = dataset.train.images.mean(0)
        dataset.train.images = (dataset.train.images - mean)

        if hasattr(dataset, 'test'):
            dataset.test.images = (dataset.test.images - mean)
        if hasattr(dataset, 'validation'):
            dataset.validation.images = (dataset.validation.images - mean)

    elif method == "Min-Max (0/1)":
        minVal = dataset.train.images.min(0)
        maxVal = dataset.train.images.max(0)

        dataset.train.images = (dataset.train.images - minVal) / (maxVal - minVal + 1e-30)
        if hasattr(dataset, 'test'):
            dataset.test.images = (dataset.test.images - minVal) / (maxVal - minVal + 1e-30)
        if hasattr(dataset, 'validation'):
            dataset.validation.images = (dataset.validation.images - minVal) / (maxVal - minVal + 1e-30)

    elif method == "Min-Max (-1/1)":
        minVal = dataset.train.images.min(0)
        maxVal = dataset.train.images.max(0)

        dataset.train.images = ((dataset.train.images - minVal) / ((maxVal - minVal + 1e-30) / 2)) - 1
        if hasattr(dataset, 'test'):
            dataset.test.images = ((dataset.test.images - minVal) / ((maxVal - minVal + 1e-30) / 2)) - 1
        if hasattr(dataset, 'validation'):
            dataset.validation.images = ((dataset.validation.images - minVal) / ((maxVal - minVal + 1e-20) / 2)) - 1

    elif method == "L1":
        norm = np.linalg.norm(dataset.train.images, ord=1)
        if norm == 0:
            norm = np.finfo(v.dtype).eps
        dataset.train.images = dataset.train.images / norm
    else:
        raise Exception("Warn - invalid normalize method = " + method)


def read_MNIST(train_dir, one_hot=True, normalization="", use_double=False):
    data_sets = DataSets()

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 0

    local_file = maybe_download(SOURCE_URL, TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)

    local_file = maybe_download(SOURCE_URL, TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)

    local_file = maybe_download(SOURCE_URL, TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)

    local_file = maybe_download(SOURCE_URL, TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.use_double = use_double
    data_sets.train = DataSet(train_images, train_labels, use_double=use_double)

    '''
    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    
    data_sets.validation = DataSet(validation_images, validation_labels, use_double=use_double)  
    '''

    data_sets.test = DataSet(test_images, test_labels, use_double=use_double)

    if normalization != "":
        normalize(data_sets, normalization)

    return data_sets


def read_CIFAR10(train_dir, one_hot=True, normalization="", use_double=False):

    train_images = None
    train_labels = None

    for i in range(5):
        fName = train_dir + '/data_batch_' + str(i + 1)
        with open(fName, 'rb') as fo:
            dt = pickle.load(fo, encoding='bytes')

            if train_images is None:
                train_images = dt[b'data']
                train_labels = np.array(dt[b'labels'])
            else:
                train_images = np.concatenate([train_images, dt[b'data']], 0)
                train_labels = np.concatenate([train_labels, np.array(dt[b'labels'])], 0)

    test_images = None
    test_labels = None

    fName = train_dir + '/test_batch'
    with open(fName, 'rb') as fo:
        dt = pickle.load(fo, encoding='bytes')
        test_images = dt[b'data']
        test_labels = np.array(dt[b'labels'])

    if one_hot:
        train_labels = dense_to_one_hot(train_labels, 10)
        test_labels = dense_to_one_hot(test_labels, 10)

    data_set = DataSets()
    data_set.train = DataSet(train_images, train_labels, to_row_vectors=False)
    data_set.test = DataSet(test_images, test_labels, to_row_vectors=False)

    data_set.width = data_set.train.width = data_set.test.width = 32
    data_set.height = data_set.train.height = data_set.test.height = 32
    data_set.channel = data_set.train.channel = data_set.test.channel = 3

    if normalization != "":
        normalize(data_set, normalization)

    return data_set


def read_CIFAR10_Stream(train_dir, one_hot=True):

    train_files = []
    for i in range(5):
        train_files.append(train_dir + '/data_batch_' + str(i + 1) + '.bin')

    test_files = [train_dir + '/test_batch.bin' ]

    data_set = DataSets()
    data_set.train = StreamDataSet(train_files, 32, 32, 3, dim_labels=10, label_bytes=1,
                                   num_examples=50000, num_examples_per_epoch=50000)
    data_set.test = StreamDataSet(test_files, 32, 32, 3, dim_labels=10, label_bytes=1,
                                  num_examples=10000, num_examples_per_epoch=50000)

    return data_set

def read_ETRI_DF(dataPath, w, h, normalization="", use_double=False):
    def read_data(name):
        data = []
        label = []

        for fName in os.listdir(dataPath + '/' + name):
            if fName.endswith('i.jpg'):
                fName = fName[0:-5]
                # colorImg = cv2.imread(dataPath + '/Training/' + fName + "c.jpg")
                maskImg = cv2.imread(dataPath + '/Training/' + fName + "i.jpg", 0)
                gtImg = cv2.imread(dataPath + '/Training/' + fName + "t.jpg", 0)

                maskImg = cv2.resize(maskImg, (w, h))
                gtImg = cv2.resize(gtImg, (w, h))
                cv2.threshold(gtImg, 10, 1, cv2.THRESH_BINARY, gtImg)

                maskImg = maskImg.reshape(h, w, 1)
                gtImg = gtImg.reshape(w * h)

                data.append(maskImg)
                label.append(gtImg)

        data = np.array(data)
        label = np.array(label)

        return data, label

    trainData, trainLabel = read_data('Training')
    testData, testLabel = read_data('Test')

    data_sets = DataSets()
    data_sets.train = DataSet(trainData, trainLabel)
    data_sets.test = DataSet(testData, testLabel)

    if normalization != "":
        normalize(data_sets, normalization)

    return data_sets


def read_FER2013(dataPath, one_hot=True, normalization='', use_double=False):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    lineNo = 0
    with open(dataPath + '/fer2013.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if lineNo > 0:
                if row[2] == 'Training':
                    train_images.append(list(map(int, row[1].split(' '))))
                    train_labels.append(int(row[0]))
                elif row[2] == 'PublicTest':
                    test_images.append(list(map(int, row[1].split(' '))))
                    test_labels.append(int(row[0]))
            lineNo += 1

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    if one_hot:
        train_labels = dense_to_one_hot(train_labels)
        test_labels = dense_to_one_hot(test_labels)

    data_set = DataSets()
    data_set.train = DataSet(train_images, train_labels, is_image=False)
    data_set.test = DataSet(test_images, test_labels, is_image=False)

    data_set.width = data_set.train.width = data_set.test.width = 48
    data_set.height = data_set.train.height = data_set.test.height = 48
    data_set.channel = data_set.train.channel = data_set.test.channel = 1

    if normalization != "":
        normalize(data_set, normalization)

    return data_set

def read_ETRI_Engagement(dataPath, one_hot=True, normalization=''):
    train_images = []
    train_labels = []

    img_width = 64
    img_height = 64

    for fileName in os.listdir(dataPath + '/Data'):
        img = cv2.imread(dataPath + '/Images/' + fileName[0:-4] + '.jpg')
        with open(dataPath + '/Data/' + fileName) as txt:
            while True:
                line = txt.readline()
                if not line: break

                args = line.split()
                if len(args) == 5:
                    x = int(args[0])
                    y = int(args[1])
                    w = int(args[2])
                    h = int(args[3])
                    l = int(args[4])

                    if l == -1 :
                        continue
                    if x < 0 or y < 0 or x + w >= img.shape[1] or y + h >= img.shape[0]:
                        continue

                    l -= 1

                    fImg = cv2.resize(img[y:y+h, x:x+w], (img_width, img_height))
                    train_images.append(fImg)
                    train_labels.append(l)


    perm = np.arange(len(train_images))
    np.random.shuffle(perm)

    for i in range(int(len(train_images) / 2)):
        idx1 = perm[i * 2]
        idx2 = perm[(i * 2) + 1]

        temp = train_images[idx1].copy()
        train_images[idx1] = train_images[idx2]
        train_images[idx2] = temp

        temp = train_labels[idx1]
        train_labels[idx1] = train_labels[idx2]
        train_labels[idx2] = temp

    test_size = 1000
    train_size = len(train_images) - test_size

    test_images = train_images[train_size:]
    test_labels = train_labels[train_size:]
    train_images = train_images[0:train_size]
    train_labels = train_labels[0:train_size]

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    if one_hot:
        train_labels = dense_to_one_hot(train_labels, num_classes=3)
        test_labels = dense_to_one_hot(test_labels, num_classes=3)

    data_set = DataSets()
    data_set.train = DataSet(train_images, train_labels)
    data_set.test = DataSet(test_images, test_labels)

    data_set.width = data_set.train.width = data_set.test.width = img_width
    data_set.height = data_set.train.height = data_set.test.height = img_height
    data_set.channel = data_set.train.channel = data_set.test.channel = 3

    if normalization != "":
        normalize(data_set, normalization)

    return data_set