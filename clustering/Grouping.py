import math
import sklearn.cluster
import cv2
import numpy as np
import os

import clustering.RankOrderClustering as ROC

import elm.Utilities as util
import elm.NaiveELM as NaiveELM
import elm.DataInput as DataInput

class GroupInfo :

    def __init__(self):
        self.features = []
        self.index = []
        self.confidence = []

    @property
    def num_examples(self):
        return len(self.index)

    @property
    def average_confidence(self):
        if len(self.confidence) > 0:
            return np.mean(self.confidence)
        else:
            return 0.0

def do_kMeans(cluster_k, images, conf, dataset):
    clusters = []

    print('Perform kMeans ...')
    util.record_time()
    km = sklearn.cluster.KMeans(n_clusters=10).fit(dataset.train.images)
    print('Done (%s)' % util.get_elapsed_time())

    for i in range(cluster_k):
        clusters.append(GroupInfo())

    for idx, c_idx in enumerate(km.labels_):
        clusters[c_idx].index.append(idx)
        clusters[c_idx].confidence.append(conf[idx])


def do_ROClustering(rank_k, cluster_k, temporal_k, conf, frameNo, dataset):
    clusters = []

    f_dim = dataset.train.dim_features
    def spatial_temporal_dist(data, i, j):
        a_dist = util.absolute_distance(data, i, j) / f_dim
        t_dist = 0

        if temporal_k > 0 :
            # t_dist = (math.pow(frameNo[i] - frameNo[j], 2) / temporal_k)
            t_dist = abs(frameNo[i] - frameNo[j])
            if t_dist > temporal_k :
                t_dist = 1
            else:
                t_dist /= temporal_k

        return a_dist + t_dist

    # print('Perform RankOrder clustering ...')
    util.record_time()
    c = ROC.RankOrderClustering(dataset.train.images, cluster_k, rank_k=15, cluster_type='RankOrder',
                                dist_func=spatial_temporal_dist)
    c.run()
    # print('Time : ' + str(util.get_elapsed_time()))
    for i in range(cluster_k):
        clusters.append(GroupInfo())

    for cluster_idx, cl in enumerate(c.clusters):
        for item_idx, item in enumerate(cl):
            clusters[cluster_idx].index.append(item)
            clusters[cluster_idx].confidence.append(conf[item])

    return clusters

def do_OutlierRemove(node_ratio, min_iter, max_iter, images):
    trainType = 0
    images = np.array(images)

    features = np.reshape(images, newshape=[images.shape[0], images.shape[1] * images.shape[2] * images.shape[3]])
    total_train_dataset = DataInput.DataSet(features, features, is_image=False)

    dataset = DataInput.DataSets()
    dataset.train = total_train_dataset

    # Discriminative Clustering
    e = NaiveELM.ELM()
    e.store_h = False
    e.use_double = False
    e.activation = 'ReLU'

    util.record_time()
    prev_idx = None
    prev_err = None

    for ti in range(max_iter) :
        e.init_weight(dataset, int(dataset.train.num_examples * node_ratio), init_method='RP')
        if trainType == 0:
            e.train(dataset)
        elif trainType == 1:
            e.train_sgd(dataset, int(len(images) * node_ratio), 1000, learning_rate=[1e-3], momentum=0,
                        w1_tune=True, display_step=100)

        # Calculate Reconstruction Error
        dataset.train = total_train_dataset
        res = e.predict(dataset.train)
        err = np.power(res - dataset.train.images, 2)
        err = np.sqrt(err)
        err = np.sum(err, axis=1)

        prev_err = err
        cv2.normalize(err, err, 0, 255, cv2.NORM_MINMAX)
        err = np.uint8(err)

        otsu_t, ret = cv2.threshold(err, 0.5, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret = ret.reshape(len(ret))
        idx_0 = np.where(ret == 0)[0]
        idx_1 = np.where(ret == 1)[0]

        # One cluster
        if len(idx_0) == 0:
            prev_idx = idx_1
            break
        elif len(idx_1) == 0 :
            prev_idx = idx_0
            break

        # Separate Pos/Neg
        if len(idx_0) > len(idx_1):
            num_pos = len(idx_0)
            pos_idx = idx_0
        else:
            num_pos = len(idx_1)
            pos_idx = idx_1

        pos_feature = np.zeros(shape=[num_pos, total_train_dataset.dim_features])
        for i in range(num_pos):
            pos_feature[i] = features[pos_idx[i]]

        if prev_idx is not None:
            if ti > min_iter and np.count_nonzero(prev_idx - ret) == 0 :
                # print('Converged at Iteration %d' % ti)
                break

        prev_idx = ret
        dataset.train = DataInput.DataSet(pos_feature, pos_feature, is_image=False)

    return prev_idx, prev_err