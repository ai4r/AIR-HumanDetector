
from sklearn.neighbors import NearestNeighbors
import numpy as np
import elm.Utilities as util

class RankOrderClustering:

    def __init__(self, features, k, rank_k=0, distance_type='L1', cluster_type='RankOrder',
                 dist_func=None):
        self.k = k
        self.rank_k = rank_k
        self.distance_type = distance_type
        self.distance_func = dist_func
        self.cluster_type = cluster_type
        self.features =features
        self.use_approximation = False

    def run(self):

        if (self.cluster_type == 'RankOrder' or self.cluster_type == 'RankMatch') and self.k <= 0 :
            raise Exception('k must be > 0')

        num_of_samples = self.features.shape[0]

        if self.k > 1:

            if not self.use_approximation :
                if self.distance_func is None:
                    pairwise_dist = util.calcDistMatrix(self.features, dist_type=self.distance_type, do_normalize=True)
                else:
                    pairwise_dist = util.calcDistMatrix(self.features, dist_func=self.distance_func, do_normalize=True)

                self.pairwise_dist = pairwise_dist
                self.pairwise_rank = np.argsort(pairwise_dist)
            else:
                nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(self.features)
                self.pairwise_dist, self.pairwise_rank = nbrs.kneighbors(self.features)

            # 1-NN sequence
            idx = 0
            nn_seq = [idx]
            marked = np.full(num_of_samples, 0, dtype=np.int32)
            marked[idx] = 0

            while(len(nn_seq) < num_of_samples):
                last_idx = idx
                for j in range(1, self.k) :
                    n_idx = self.pairwise_rank[idx][j]
                    if marked[n_idx] == 0:
                        nn_seq.append(n_idx)
                        marked[n_idx] = 1
                        idx = n_idx
                        break

                if last_idx == idx:
                    for j in range(0, num_of_samples):
                        if marked[j] == 0:
                            marked[j] = 1
                            nn_seq.append(idx)
                            idx = j
                            break

            dist_seq = []
            for i in range(1, len(nn_seq)) :
                idx1 = nn_seq[i-1]
                idx2 = nn_seq[i]

                if self.cluster_type == 'Pairwise' :
                    dist = self.pairwise_dist[idx1][idx2]

                elif self.cluster_type == 'RankOrder' or self.cluster_type == 'RankMatch':
                    dist = self.calcRankOderDist(idx1, idx2)

                else:
                    raise Exception('Invalid cluster_type = ' + str(self.cluster_type))
                dist_seq.append(dist)

            clusters = []
            min_idx = np.argsort(dist_seq)[::-1]
            cut_points = [0]
            for i in range(self.k-1):
                cut_points.append(min_idx[i]+1)
            cut_points.sort()
            cut_points.append(len(nn_seq))

            for i in range(1, len(cut_points)):
                st = cut_points[i-1]
                ed = cut_points[i]
                clusters.append(nn_seq[st:ed])
        else:
            c = []
            for i in range(num_of_samples):
                c.append(i)
            clusters = [c]

        self.clusters = clusters
        self.cluster_index = np.zeros(num_of_samples, dtype=np.int32)
        for cluster_idx, cl in enumerate(clusters):
            for item_idx in cl:
                self.cluster_index[item_idx] = cluster_idx

    def get_rank(self, i, j):
        idx = np.where(self.pairwise_rank[i] == j)[0]

        if len(idx) == 0:
            return self.k + 1
        else:
            return idx[0]

    def get_rank_sum(self, i, j, k):
        sum = 0
        nn_i = self.pairwise_rank[i]
        nn_j = self.pairwise_rank[j]

        if k <= 0:
            k = self.pairwise_rank.shape[1]

        for ni in range(len(nn_j)):
            idx = np.where(nn_j == nn_i[ni])[0]

            if len(idx) == 0 :
                idx = self.k + 1
            else:
                idx = idx[0]

            if self.cluster_type == 'RankOrder':
                sum += idx
            elif self.cluster_type == 'RankMatch':
                if idx > k:
                    sum += 1
            else:
                raise Exception('Invalid type = ' + str(type))
            if ni > k:
                break
        return sum


    def calcRankOderDist(self, i, j):
        if i == j:
            return 0
        else:
            o_a = self.get_rank(i, j)
            o_b = self.get_rank(j, i)
            d_a = self.get_rank_sum(i, j, self.rank_k)
            d_b = self.get_rank_sum(j, i, self.rank_k)

            dist = d_a + d_b

            if o_a == 0 or o_b == 0:
                norm_factor = 1
            elif o_a > o_b:
                norm_factor = o_b
            else:
                norm_factor = o_a

            dist /= norm_factor
            return dist