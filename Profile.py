"""
Profile of tracked object
Classifier: SVM
"""

import cv2
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


MAX_KEEP_IMAGE_COUNT = 100
PROFILE_ASSIGN_DISTANCE_THRESHOLD = 0.5
PROFILE_ASSIGN_SVM_THRESHOLD = 0.5

CLASSIFIER_DISTANCE_NEGATIVE_RATIO = 1.5
MAX_KEEP_PROFILE = 3





class ProfileClassifier:
    """
    Determine which profile to assign
    """
    def __init__(self, classifierType = 'SVC'):
        # list of profile
        self.profile_list = []
        # svm classifier
        if classifierType == 'SVC':
            self.clf = svm.SVC(kernel='linear', probability=True)
        elif classifierType == 'KNC':
            self.clf = KNeighborsClassifier(MAX_KEEP_PROFILE)

    def train_svm(self):
        if len(self.profile_list) > 1:
            x_train = []
            y_train = []
            for each_profile in self.profile_list:
                uid = each_profile.uid
                feature_origin_list = each_profile.conf_feature_list
                for each_feature in feature_origin_list:
                    x_train.append(each_feature[0])
                    y_train.append(uid)

            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)

            self.clf.fit(x_train, y_train)

    def test_svm(self, current_feature):
        res = self.clf.predict_proba(current_feature)
        return res

    def update_profile_with_uid(self, uid, current_feature, current_image, current_rect, tracking_confidence):
        if(uid >= 0):
            found_profile = self.get_profile_with_uid(uid)
            found_profile.update_feature_list(current_feature, current_image, current_rect, tracking_confidence)

            # train svm
            self.train_svm()


    def get_profile_with_uid(self, uid):
        for each_profile in self.profile_list:
            if each_profile.uid == uid:
                return each_profile


    def register_profile(self, uid, current_feature, current_image, current_rect, tracking_confidence):
        if len(self.profile_list) <= MAX_KEEP_PROFILE:
            new_profile = Profile(uid)
            new_profile.update_feature_list(current_feature, current_image, current_rect, tracking_confidence)
            self.profile_list.append(new_profile)
        else:
            print("Exceed the maximum profile keep length...")

    def classify_profile_svm(self, current_feature):
        if len(self.profile_list) == 0:
            return []
        self.train_svm()
        current_feature_resh = np.reshape(current_feature, (1, -1))
        res = self.test_svm(current_feature_resh)
        idx_list = [i for i in range(len(self.profile_list))]

        res_list = []
        for i in range(len(idx_list)):
            res_list.append((res[0][i], idx_list[i]))

        res_list = sorted(res_list, key=lambda l: l[0], reverse=True)

        if res_list[0][0] > PROFILE_ASSIGN_SVM_THRESHOLD:
            return res_list
        else:
            return []





    def classify_profile(self, current_feature):
        # extract avg distance
        if len(self.profile_list) == 0:
            return []
        elif len(self.profile_list) == 1:
            avg_dist_list = []
            tmp_dist_list = []
            for each_profile in self.profile_list:
                avg_dist_list.append((each_profile.uid, each_profile.get_distance(current_feature)))
                #tmp_dist_list.append((each_profile.uid, each_profile.get_knn_distance(current_feature)))
                #avg_dist_list.append((each_profile.uid, each_profile.get_knn_distance(current_feature)))

            # extract minimum distance
            min_dist = 9999
            min_uid = -1

            for each_avg_dist in avg_dist_list:
                if min_dist > each_avg_dist[1]:
                    min_dist = each_avg_dist[1]
                    min_uid = each_avg_dist[0]

            # extract minimum distance ratio
            assign_success = True
            for each_avg_dist in avg_dist_list:
                if min_dist * CLASSIFIER_DISTANCE_NEGATIVE_RATIO < each_avg_dist[1] or min_dist > PROFILE_ASSIGN_DISTANCE_THRESHOLD:
                    assign_success = False

            # return assigned uid
            if assign_success:
                return [min_uid]
            # fail to found uid; -1 --> register new object
            else:
                return []
        else:
            return self.classify_profile_svm(current_feature)
            """
            avg_dist_list = []
            tmp_dist_list = []
            for each_profile in self.profile_list:
                #avg_dist_list.append((each_profile.uid, each_profile.get_distance(current_feature)))
                #avg_dist_list.append((each_profile.uid, each_profile.get_distance(current_feature)))
                avg_dist_list.append((each_profile.uid, each_profile.get_knn_distance(current_feature)))

            # extract minimum distance
            min_dist = 9999
            min_uid = -1

            for each_avg_dist in avg_dist_list:
                if min_dist > each_avg_dist[1]:
                    min_dist = each_avg_dist[1]
                    min_uid = each_avg_dist[0]

            # extract minimum distance ratio
            assign_success = True
            for each_avg_dist in avg_dist_list:
                if min_dist * CLASSIFIER_DISTANCE_NEGATIVE_RATIO < each_avg_dist[
                    1] or min_dist > PROFILE_ASSIGN_DISTANCE_THRESHOLD:
                    assign_success = False

            # return assigned uid
            if assign_success:
                return min_uid
            # fail to found uid; -1 --> register new object
            else:
                return -1
            """



class Profile:
    """
    Profile that contains the feature of tracked object
    """
    def __init__(self, uid):
        # unique id
        self.uid = uid
        # list of positive image and rect
        self.rect_image_list = []

        # list of positive confidence and feature Tuple(feature, confidence)
        self.conf_feature_list = []
        # center point
        self.center_point = None
        # distance type
        self.dist_type = cv2.HISTCMP_BHATTACHARYYA

    def get_knn_distance(self, current_feature):
        """
        Get the knn distance
        """
        dist_center = cv2.compareHist(current_feature, self.center_point, self.dist_type)
        knn_cnt = 0

        for each_conf_feature in self.conf_feature_list:
            each_feature = each_conf_feature[0]
            dist = cv2.compareHist(self.center_point, each_feature, self.dist_type)
            if dist > dist_center:
                knn_cnt+=1

        avg_dist = float(knn_cnt) / len(self.conf_feature_list)

        return avg_dist

    def get_distance(self, current_feature):
        """
        Get the average distance from the current_feature to the saved list of feature
        """
        avg_dist = 0
        for each_conf_feature in self.conf_feature_list:
            each_feature = each_conf_feature[0]
            avg_dist += cv2.compareHist(current_feature, each_feature,  self.dist_type)
        avg_dist = float(avg_dist) / len(self.conf_feature_list)

        return avg_dist

    def update_center_point(self):
        """
        Update the center point of feature list
        """
        feature_sum_vector = None
        for each_conf_feature in self.conf_feature_list:
            each_feature = each_conf_feature[0]
            if feature_sum_vector is None:
                feature_sum_vector = np.zeros(np.shape(each_feature), dtype=each_feature.dtype)
            feature_sum_vector = feature_sum_vector + each_feature

        feature_sum_vector = feature_sum_vector / len(self.conf_feature_list)

        self.center_point = feature_sum_vector

    def update_feature_list(self, current_feature, current_image, current_rect, tracking_confidence):
        """
        Update the feature list if the current_feature is better than others
        """

        if len(self.conf_feature_list) < MAX_KEEP_IMAGE_COUNT:
            self.conf_feature_list.append([current_feature, tracking_confidence])
            self.rect_image_list.append([current_image, current_rect])
            # update the center point
            self.update_center_point()

            return True
        """
        # 1) filter by tracking confidence

        # get the minimum confidence
        is_update = False
        min_conf = 99999
        min_idx = -1
        for idx, each_conf_feature in enumerate(self.conf_feature_list):
            each_conf = each_conf_feature[1]
            if min_conf > each_conf:
                min_conf = each_conf
                min_idx = idx

        if min_conf < tracking_confidence:
            is_update = True
        else:
            is_update = False

        # 2) distance between the center point and the current feature

        if is_update:
            max_dist = -9999
            max_idx = -1
            for idx, each_conf_feature in enumerate(self.conf_feature_list):
                each_feature = each_conf_feature[0]
                dist = cv2.compareHist(self.center_point, each_feature, self.dist_type)

                if max_dist < dist:
                    max_dist = dist
                    max_idx = idx

            dist = cv2.compareHist(self.center_point, current_feature, self.dist_type)

            if max_dist > dist:
                is_update = True
            else:
                is_update = False

        # finally update the feature
        # Update the idx that far away from the center point
        if is_update:
            self.conf_feature_list[max_idx][0] = current_feature
            self.conf_feature_list[max_idx][1] = tracking_confidence
            self.rect_image_list[max_idx][0] = current_image
            self.rect_image_list[max_idx][1] = current_rect
        
        """

        # update center point
        self.update_center_point()












