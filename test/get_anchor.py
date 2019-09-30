# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import random
import sklearn.cluster as cluster
import xml.etree.ElementTree as XMLTree


def iou(x, centroids):
    dists = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            dist = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            dist = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            dist = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            dist = (c_w * c_h) / (w * h)
        dists.append(dist)
    return np.array(dists)


def avg_iou(x, centroids):
    n, d = x.shape
    sums = 0.
    for i in range(x.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i]
        # slightly ineffective, but I am too lazy
        sums += max(iou(x[i], centroids))
    return sums / n


def write_anchors_to_file(centroids, distance, img_width, grid_count):
    anchors = centroids * (img_width / grid_count)
    anchors = [str(i) for i in anchors.ravel()]
    print(
        "\n",
        "Cluster Result:\n",
        "Clusters:", len(centroids), "\n",
        "Average IoU:", distance, "\n",
        "Anchors:\n",
        ", ".join(anchors)
    )
    print(", ".join(anchors))

    '''
    with open(anchor_file, 'w') as f:
        f.write(", ".join(anchors))
        f.write('\n%f\n' % distance)
    '''


def k_means(x, n_clusters, eps):
    init_index = [random.randrange(x.shape[0]) for _ in range(n_clusters)]
    centroids = x[init_index]

    d = old_d = []
    iterations = 0
    diff = 1e10
    c, dim = centroids.shape

    while True:
        iterations += 1
        d = np.array([1 - iou(i, centroids) for i in x])
        if len(old_d) > 0:
            diff = np.sum(np.abs(d - old_d))

        print('diff = %f' % diff)

        if diff < eps or iterations > 1000:
            print("Number of iterations took = %d" % iterations)
            print("Centroids = ", centroids)
            return centroids

        # assign samples to centroids
        belonging_centroids = np.argmin(d, axis=1)

        # calculate the new centroids
        centroid_sums = np.zeros((c, dim), np.float)
        for i in range(belonging_centroids.shape[0]):
            centroid_sums[belonging_centroids[i]] += x[i]

        for j in range(c):
            centroids[j] = centroid_sums[j] / np.sum(belonging_centroids == j)

        old_d = d.copy()


def get_file_content(fnm):
    with open(fnm) as f:
        return [line.strip() for line in f]


def get_data(dir, target):
    X = []
    annotation_dir = dir + '/Annotations'
    file_list = os.listdir(annotation_dir)
    for idx, filePath in enumerate(file_list):
        tree = XMLTree.parse('%s/%s' % (annotation_dir, filePath))
        root = tree.getroot()

        if idx % 1000 == 0:
            print('   %d files processed...' % idx)

        w = int(root.find('size').find('width').text)
        h = int(root.find('size').find('height').text)

        objs = root.findall('object')
        for obj in objs:
            if target is not '' and not obj.find('name').text == target :
                continue

            bndbox = obj.find('bndbox')

            x1 = round(float(bndbox.find('xmin').text))
            x2 = round(float(bndbox.find('xmax').text))
            y1 = round(float(bndbox.find('ymin').text))
            y2 = round(float(bndbox.find('ymax').text))

            normalized_w = (x2 - x1) / w
            normalized_h = (y2 - y1) / h

            if normalized_w <= 0 or normalized_h <= 0 or w <= 0 or h <= 0 :
                print('Error! %s - %d, %d, %d, %d' % (filePath, x1, x2, y1, y2))
                continue


            X.append([normalized_w, normalized_h])
            # X.append([x2 - x1, y2 - y1])

    return np.array(X)

k = 10
target = ''
grid_count = 19
img_width = 608
dir = '/media/data/PTMP/HumanCare/Sitting+VOC+Walking'
method = 'km(iou)'
tol = 1e-20
run_count = 5

print('Loading data from %s' % dir)
data = get_data(dir, target)
print('Done. %d samples' % len(data))


results = []
for i in range(run_count):
    if method == 'km':
        km = cluster.KMeans(n_clusters=k, tol=tol, verbose=True)
        km.fit(data)
        result = km.cluster_centers_

    elif method == 'mini-km':
        km = cluster.MiniBatchKMeans(n_clusters=k, tol=tol, verbose=True)
        km.fit(data)
        result = km.cluster_centers_

    elif method == 'km(iou)':
        result = k_means(data, k, tol)

    else:
        raise Exception('Invalid method = ' + method)

    result = result[result[:, 0].argsort()]
    results.append(result)

print('Average Centroids -----------------')
results = np.array(results)
results = np.mean(results, axis=0)
distance = avg_iou(data, results)
print(results)

print('After Multiplication -----------------')
results = results * (grid_count)
results = np.round(results, 5)
print(results)

print('Average IOU : ' + str(distance))
results = [str(i) for i in results.ravel()]
print(", ".join(results))
