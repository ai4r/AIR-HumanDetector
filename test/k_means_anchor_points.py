# -*- coding: utf-8 -*-
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

# Original code @ferada http://codereview.stackexchange.com/questions/128315/k-means-clustering-algorithm-implementation

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

    for i in range(len(new_centroids)):
        shift = niou(centroids[i], new_centroids[i])
        print("Cluster {} size: {}".format(i, len(clusters[i])))
        print("Centroid {} distance shift: {}\n\n".format(i, shift))

    if iter_count < best_avg_iou_iteration + iteration_cutoff:
        kmeans(k, new_centroids, points, iter_count+1, iteration_cutoff)

    return

def plot_anchors(pascal_anchors, coco_anchors):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim([0,500])
    ax1.set_xlim([0,900])

    for i in range(len(pascal_anchors)):
        if area(pascal_anchors[i]) > area(coco_anchors[i]):
            bbox1 = pascal_anchors[i]
            color1 = "white"
            bbox2 = coco_anchors[i]
            color2 = "blue"
        else:
            bbox1 = coco_anchors[i]
            color1 = "blue"
            bbox2 = pascal_anchors[i]
            color2 = "white"

        lower_right_x = bbox1[0]-(bbox1[2]/2.0)
        lower_right_y = bbox1[1]-(bbox1[3]/2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_right_x, lower_right_y),   # (x,y)
                bbox1[2],          # width
                bbox1[3],          # height
                facecolor=color1
            )
        )

        lower_right_x = bbox2[0]-(bbox2[2]/2.0)
        lower_right_y = bbox2[1]-(bbox2[3]/2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_right_x, lower_right_y),   # (x,y)
                bbox2[2],          # width
                bbox2[3],          # height
                facecolor=color2
            )
        )
    plt.show()


if __name__ == "__main__":
    # Load pascal and coco label data (original coordinates)
    # shape: [[x1,y1,w1,h1],...,[xn,yn,wn,hn]]
    pascal_data = []
    coco_data = []

    with open("2007_train.txt", 'r') as f:
        for line in f:
            line = line.replace("JPEGImages", "labels_original")
            line = line.replace(".jpg", ".txt").strip()

            with open(line, 'r') as l:
                for line2 in l:
                    bbox_list = line2.split(" ")[1:]
                    bbox_list = [float(x.strip()) for x in bbox_list]
                    pascal_data.append(bbox_list)


    coco_label_paths = glob.glob("coco_labels_original/*.txt")
    for p in coco_label_paths:
        with open(p, 'r') as l:
            for line in l:
                bbox_list = line.split(" ")[1:]
                bbox_list = [float(x.strip()) for x in bbox_list]
                if len(bbox_list) != 4:
                    continue
                coco_data.append(bbox_list)


    pascal_data = np.array(pascal_data)
    coco_data = np.array(coco_data)

    # Set x,y coordinates to origin
    for i in range(len(pascal_data)):
        pascal_data[i][0] = 0
        pascal_data[i][1] = 0

    for i in range(len(coco_data)):
        coco_data[i][0] = 0
        coco_data[i][1] = 0

    # k-means picking the first k points as centroids
    k = 5
    centroids = pascal_data[:k]
    kmeans(k, centroids, pascal_data)

    # Get anchor boxes from best clusters
    pascal_anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])

    # Sort by width
    pascal_anchors = pascal_anchors[pascal_anchors[:, 2].argsort()]

    # scaled pascal anchors from cfg (for comparison): 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    print("\nk-means clustering pascal anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, pascal_anchors))

    best_avg_iou_iteration = 0
    best_avg_iou = 0

    centroids = coco_data[:k]
    kmeans(k, centroids, coco_data)

     # Get anchor boxes from best clusters
    coco_anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])

    # Sort by width
    coco_anchors = coco_anchors[coco_anchors[:, 2].argsort()]

    # scaled coco anchors from cfg (for comparison): 0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741
    print("\nk-means clustering coco anchor points (original coordinates) \
    \nFound at iteration {} with best average IoU: {} \
    \n{}".format(best_avg_iou_iteration, best_avg_iou, coco_anchors))

    # # Hardcoded results to skip computation
    # pascal_anchors = np.asarray(
    # [[   0.,            0.,          295.43055556,  157.86944444],
    #  [   0.,            0.,           45.14861187,   62.17800762],
    #  [   0.,            0.,          361.03531786,  323.51160444],
    #  [   0.,            0.,          160.96848934,  267.35032437],
    #  [   0.,            0.,          103.46714456,  131.91278375]])
    #
    # coco_anchors = np.asarray(
    # [[   0.,            0.,          159.01092483,  118.88467982],
    #  [   0.,            0.,           64.93587645,   58.59559227],
    #  [   0.,            0.,          475.61739025,  332.0915918 ],
    #  [   0.,            0.,          195.32259936,  272.98913297],
    #  [   0.,            0.,           19.46456615,   21.47707798]])
    # pascal_anchors = pascal_anchors[pascal_anchors[:, 2].argsort()]
    # coco_anchors = coco_anchors[coco_anchors[:, 2].argsort()]

    # Hardcode anchor center coordinates for plot
    pascal_anchors[3][0] = 250
    pascal_anchors[3][1] = 100
    pascal_anchors[0][0] = 300
    pascal_anchors[0][1] = 450
    pascal_anchors[4][0] = 650
    pascal_anchors[4][1] = 250
    pascal_anchors[2][0] = 110
    pascal_anchors[2][1] = 350
    pascal_anchors[1][0] = 300
    pascal_anchors[1][1] = 300

    # # Reorder centroid 2 and 5 in coco anchors
    tmp = np.copy(coco_anchors[2])
    coco_anchors[2] = coco_anchors[3]
    coco_anchors[3] = tmp

    coco_anchors[3][0] = 250
    coco_anchors[3][1] = 100
    coco_anchors[0][0] = 300
    coco_anchors[0][1] = 450
    coco_anchors[4][0] = 650
    coco_anchors[4][1] = 250
    coco_anchors[2][0] = 110
    coco_anchors[2][1] = 350
    coco_anchors[1][0] = 300
    coco_anchors[1][1] = 300

    plot_anchors(pascal_anchors, coco_anchors)
