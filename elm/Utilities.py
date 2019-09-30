import time
import csv
import numpy as np
import cv2

from multiprocessing.dummy import Pool as ThreadPool

___recorded_time = [0, 0, 0, 0, 0]

def record_time(idx=0):
    global ___recorded_time
    ___recorded_time[idx] = time.time()


def get_elapsed_time(idx=0, sec_unit=False):
    diff = time.time() - ___recorded_time[idx]
    if sec_unit:
        return diff
    else:
        return get_readable_time_string(diff)

def get_readable_time_string(diff):
    m = int(diff / 60)
    h = int(m / 60)
    d = int(h / 24)

    s = diff % 60
    m = m % 60
    h = h % 24

    duration = "%ds" % s

    if m > 0:
        duration = "%dm %s" % (m, duration)
    if h > 0:
        duration = "%dh %s" % (h, duration)
    if d > 0:
        duration = "%dd %s" % (d, duration)

    return duration


def log(msg):
    current_time = time.strftime("%y%m%d %H:%M:%S", time.localtime())
    print("[%s] %s" % (current_time, msg))


def save_csv(data_list, filename, is_append=False):
    if is_append:
        csvfile = open(filename, 'a')
    else:
        csvfile = open(filename, 'w')

    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(data_list)):
        writer.writerow(data_list[i])
    csvfile.close()

def blockwise_inverse(M, rcond=1e-15):
    M = M.astype(np.float64)

    n_rows = M.shape[0]
    n_cols = M.shape[1]

    A_rows = int(n_rows / 2)
    A_cols = int(n_cols / 2)

    A = M[0:A_rows, 0:A_cols]
    B = M[A_rows:n_rows, A_cols:n_cols]
    C = M[A_rows:n_rows, 0:A_cols]
    D = M[A_rows:n_rows, A_cols:n_cols]

    D_inv = np.linalg.pinv(D, rcond=rcond)

    BD_inv = np.matmul(B, D_inv)
    D_inv_C = np.matmul(D_inv, C)

    E = np.matmul(BD_inv, C)
    E = A - E
    E = np.linalg.pinv(E, rcond=rcond)

    F = np.matmul(-E, BD_inv)
    G = np.matmul(-D_inv_C, E)

    H = np.matmul(D_inv_C, E)
    H = np.matmul(H, BD_inv)
    H = H + D_inv

    P1 = np.concatenate([E, F], axis=1)
    P2 = np.concatenate([G, H], axis=1)

    P = np.concatenate([P1, P2], axis=0)

    P = P.astype(np.float32)
    return P

def vectors_to_image(data, width, height, channel, do_normalize=True):
    images = []

    sp = np.shape(data)
    img = None
    for ii in range(sp[0]):
        if channel == 1 :
            img = data[ii].reshape(height, width, 1)
            if do_normalize:
                cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

        elif channel == 3:
            per_length = int(sp[1] / channel)
            sub_images = []
            for ci in range(channel):
                img = data[ii][(ci * per_length) : (per_length * (ci+1))]
                if do_normalize:
                    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
                img = img.reshape(height, width, 1)
                sub_images.append(img)
            img = cv2.merge(sub_images)

        else:
            raise Exception('channel must be 1 or 3! not %d' % (channel))

        if do_normalize:
            img = img.astype(np.uint8)
        images.append(img)

    return images

def crop(src, rect, pt1=None, pt2=None):
    if pt1 is None or pt2 is None:
        x, y, w, h = rect
        return src[y:y+h, x:x+w]
    else:
        return src[pt1[1]:pt2[1], pt1[0]:pt2[0]]

def is_in_rect(target, rect):
    if len(target) == 2:
        return target[0] > 0 and target[0] < (rect[0] + rect[2]) and target[1] > 0 and target[1] < (rect[1] + rect[3])
    if len(target) == 4:
        return target[0] > rect[0] and (target[0] + target[2]) < (rect[0] + rect[2]) and \
               target[1] > rect[1] and (target[1] + target[3]) < (rect[1] + rect[3])

def expand_rect(img, rect, w_ratio, h_ratio):
    nw = int(rect[2] * w_ratio)
    nh = int(rect[3] * h_ratio)
    dw = nw - rect[2]
    dh = nh - rect[3]
    nx = int(rect[0] - dw / 2)
    ny = int(rect[1] - dh / 2)

    if nx < 0:
        nx = 0
    if ny < 0:
        ny = 0
    if nx + nw > img.shape[1]:
        nw = img.shape[1] - nx
    if ny + nh > img.shape[0]:
        nh = img.shape[0] - ny

    return (nx, ny, nw, nh)

def getOverlapRatio(pt1, pt2, is_rect=False):
    if is_rect:
        A_left = pt1[0]
        A_top = pt1[1]
        A_right = pt1[0] + pt1[2]
        A_bottom = pt1[1] + pt1[3]

        B_left = pt2[0]
        B_top = pt2[1]
        B_right = pt2[0] + pt2[2]
        B_bottom = pt2[1] + pt2[3]
    else:
        A_right = pt1[2]
        A_left = pt1[0]
        A_top = pt1[1]
        A_bottom = pt1[3]

        B_right = pt2[2]
        B_left = pt2[0]
        B_top = pt2[1]
        B_bottom = pt2[3]

    A_height = A_bottom - A_top
    A_width = A_right - A_left
    B_height = B_bottom - B_top
    B_width = B_right - B_left

    if (A_right < B_left) or (A_left > B_right) or (A_bottom < B_top) or (A_top > B_bottom):
        return 0.0

    width = min(A_right, B_right) - max(A_left, B_left)
    height = min(A_bottom, B_bottom) - max(A_top, B_top)

    return float(width * height) / ((A_width * A_height) + (B_width * B_height) - (width * height))


def sort(value, target, isAscending=True, isOneHot=False) :
    if isOneHot :
        idx = np.argmax(value, axis=1)
    else:
        idx = value

    sorted_idx = np.argsort(idx)

    if not isAscending:
        sorted_idx = sorted_idx[::-1]

    for i in range(len(target)):
        target[i] = target[i][sorted_idx]
    return target

def equalizeHist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def indentXML(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indentXML(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def calcDistMatrix(data, dist_type='L2', dist_func=None, do_normalize=False):
    dist = np.zeros(shape=[data.shape[0], data.shape[0]], dtype=np.float32)

    if dist_func is None:
        if dist_type == 'L1':
            dist_func = absolute_distance
        elif dist_type == 'L2':
            dist_func = euclidian_distance
        else:
            raise Exception('Invalid dist_type = %s' % (str(dist_type)))

    def calc_pair(ii):
        for ij in range(ii+1, data.shape[0]):
            v = dist_func(data, ii, ij)
            dist[ii][ij] = dist[ij][ii] = v

    p = ThreadPool(40)
    p.map(calc_pair, range(data.shape[0]))
    p.close()
    p.join()

    if do_normalize:
        dist_max = np.amax(dist)
        if dist_max > 0 :
            dist /= dist_max

    return dist

def calcRankOrderDistMatrix(distMat, k=10, rankMat=None, type='Order', do_normalize=False) :
    roDist = np.zeros(shape=[distMat.shape[0], distMat.shape[0]], dtype=np.float32)

    if rankMat is None:
        rankMat = np.argsort(distMat)

    def get_rank(i, j):
        idx = np.where(rankMat[i] == j)
        return idx[0][0]

    def get_rank_sum(i, j):
        sum = 0

        nn_i = rankMat[i]
        nn_j = rankMat[j]
        for ni in range(len(nn_j)):
            idx = np.where(nn_j == nn_i[ni])

            if type == 'Order':
                sum += idx[0][0]
            elif type == 'Match' :
                if idx[0][0] > k:
                    sum += 1
            else:
                raise Exception('Invalid type = ' + str(type))

            if ni > k:
                break

        return sum

    def calc_pair(ii):
        for ij in range(distMat.shape[0]):
            if ii == ij:
                roDist[ii][ij] = 0
            else:
                o_a = get_rank(ii, ij)
                o_b = get_rank(ij, ii)
                d_a = get_rank_sum(ii, ij)
                d_b = get_rank_sum(ij, ii)

                dist = d_a + d_b
                if o_a > o_b:
                    dist /= o_b
                else:
                    dist /= o_a

                roDist[ii][ij] = dist

    p = ThreadPool(50)
    p.map(calc_pair, range(distMat.shape[0]))
    p.close()
    p.join()

    if do_normalize:
        roDist /= np.amax(roDist)

    return roDist, rankMat


def sortAndSplit(distMat, rankMat=None, threshold=0):
    nn_indices = []
    nn_index = []

    idx = 0
    marked = np.zeros(distMat.shape[0])
    marked[idx] = 1
    marked_count = 1

    nn_index.append(idx)
    nn_indices.append(nn_index)

    if rankMat is None:
        rankMat = np.argsort(distMat)

    num_samples = distMat.shape[0]
    while marked_count < num_samples:
        for j in range(1, num_samples):
            n_idx = rankMat[idx][j]
            if marked[n_idx] == 0 :
                marked[n_idx] = 1
                pair_dist = distMat[idx][n_idx]
                print('%d-%d : %.4f' % (idx, n_idx, pair_dist))

                # Split
                if pair_dist > threshold > 0:
                    nn_indices.append(nn_index)
                    nn_index = [n_idx]
                else:
                    nn_index.append(n_idx)
                idx = n_idx
                marked_count += 1
                break

    return nn_indices

def euclidian_distance(data, i, j):
    x1 = data[i]
    x2 = data[j]
    dist = np.power(x1 - x2, 2)
    dist = np.sqrt(dist)
    dist = np.sum(dist)
    return dist

def absolute_distance(data, i, j):
    x1 = data[i]
    x2 = data[j]
    dist = np.abs(x1 - x2)
    dist = np.sum(dist)
    return dist

def shuffle(data_list):
    num_samples = data_list[0].shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)

    for i in range(int(num_samples / 2)):
        idx1 = perm[i * 2]
        idx2 = perm[(i * 2) + 1]

        for j in range(len(data_list)):
            temp = data_list[j][idx1].copy()
            data_list[j][idx1] = data_list[j][idx2]
            data_list[j][idx2] = temp

class Parameters:
    pass