# -*- coding: utf-8 -*-

import time
import numpy as np
import os
from random import randint
import xml.etree.ElementTree as XMLTree
import clustering.RankOrderClustering as ROC

best_clusters = []
best_avg_iou = 0
best_avg_iou_iteration = 0

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2.
    l2 = x2 - w2/2.
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2.
    r2 = x2 + w2/2.
    right = r1 if r1 < r2 else r2
    return right - left

def intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    return w*h

def area(x):
    return x[2]*x[3]

def union(a, b):
    return area(a) + area(b) - intersection(a, b)

def iou(a, b):
    return intersection(a, b) / union(a, b)

def niou(a, b):
    return 1. - iou(a,b)

def equals(points1, points2):
    if len(points1) != len(points2):
        return False

    for point1, point2 in zip(points1, points2):
        if point1[0] != point2[0] or point1[1] != point2[1] or point1[2] != point2[2] or point1[3] != point2[3]:
            return False

    return True

def compute_centroids(clusters):
    return [np.mean(cluster, axis=0) for cluster in clusters]

def closest_centroid(point, centroids):
    min_distance = float('inf')
    belongs_to_cluster = None
    for j, centroid in enumerate(centroids):
        dist = niou(point, centroid)

        if dist < min_distance:
            min_distance = dist
            belongs_to_cluster = j

    return belongs_to_cluster, min_distance

def kmeans(k, centroids, points, iter_count=0, iteration_cutoff=25):

    global best_clusters
    global best_avg_iou
    global best_avg_iou_iteration

    clusters = [[] for _ in range(k)]
    clusters_iou = []
    clusters_niou = []

    for point in points:
        idx, dist = closest_centroid(point, centroids)
        clusters[idx].append(point)
        clusters_niou.append(dist)
        clusters_iou.append(1.-dist)

    avg_iou = np.mean(clusters_iou)
    if avg_iou > best_avg_iou:
        best_avg_iou = avg_iou
        best_clusters = clusters
        best_avg_iou_iteration = iter_count

    print("Iteration {}".format(iter_count))
    print("Average iou to closest centroid = {}".format(avg_iou))
    print("Sum of all distances (cost) = {}\n".format(np.sum(clusters_niou)))

    new_centroids = compute_centroids(clusters)

    '''
    for i in range(len(new_centroids)):
        shift = niou(centroids[i], new_centroids[i])
        print("Cluster {} size: {}".format(i, len(clusters[i])))
        print("Centroid {} distance shift: {}\n\n".format(i, shift))
    '''

    if iter_count < best_avg_iou_iteration + iteration_cutoff:
        kmeans(k, new_centroids, points, iter_count+1, iteration_cutoff)

    return np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])


class Test:
    @staticmethod
    def get_data(dir, target):
        X = []
        annotation_dir = dir + '/Annotations'
        file_list = os.listdir(annotation_dir)
        for idx, filePath in enumerate(file_list):
            tree = XMLTree.parse('%s/%s' % (annotation_dir, filePath))
            root = tree.getroot()

            w = int(root.find('size').find('width').text)
            h = int(root.find('size').find('height').text)

            objs = root.findall('object')
            for obj in objs:
                if target is not '' and not obj.find('name').text == target :
                    continue

                bndbox = obj.find('bndbox')

                x1 = int(bndbox.find('xmin').text)
                x2 = int(bndbox.find('xmax').text)
                y1 = int(bndbox.find('ymin').text)
                y2 = int(bndbox.find('ymax').text)

                normalized_w = (x2 - x1) / w
                normalized_h = (y2 - y1) / h

                X.append([0, 0, normalized_w, normalized_h])

        print('%d samples' % len(X))
        return np.array(X)
    
    @staticmethod
    def do_kmeans(dir, k, target, iter):
        X = Test.get_data(dir, target)
        return kmeans(k, X[:k], X, iteration_cutoff=iter)

    @staticmethod
    def do_ROC(dir, k, target):
        rank_k = 10

        data = Test.get_data(dir, target)
        c = ROC.RankOrderClustering(data, k, rank_k=rank_k, cluster_type='RankOrder')
        c.use_approximation = True
        c.run()

        means = []

        for cluster_idx, cl in enumerate(c.clusters):
            sum_list = []
            for item_idx, item in enumerate(cl):
                sum_list.append(data[item])
            m = np.around(np.mean(sum_list, axis=0))
            means.append(m)

        return np.array(means)


def main():
    # print "overlap: %s" % Test.overlap()
    # print "intersection: %f" % Test.intersection()
    # print "area: %f" % Test.area()
    # print "union: %f" % Test.union()
    # print "iou: %f" % Test.iou()
    # print "uoi: %f" % Test.uoi()

    grid_count = 19
    img_width = 608
    k = 20
    target = ''
    dir = '/media/data/PTMP/HumanCare/Sitting+VOC+INRIA+Walking+YLF'

    means = Test.do_kmeans(dir, k, target, 25)
    # means = Test.do_ROC(dir, img_width, k, target)
    means = means[means[:, 2].argsort()]

    print(dir)
    print("output format is:\n\t[[x1 y1, w1, h1], ..., [xn yn, wn, hn]]")
    print("\nclustering with original dimensions:\n%s" % means)

    means *= (img_width / grid_count)
    print("\nclustering considering [%dx%d, %dx%d => %d:\n%s"\
          % (img_width, img_width, grid_count, grid_count, img_width / grid_count, means))

    for i in range(k):
        print('%f, %f,' % (means[i][2], means[i][3]), end=' ')

if __name__ == "__main__":
    main()