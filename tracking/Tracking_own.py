import math
import cv2
import numpy as np
import sys
sys.path.append("../elm")
import elm.Utilities as util



from openvino.vino_reidentification_module import get_distance_between_embeddings

import Profile

MAX_DELTA = (0.5, 0.5)
MOVE_DELTA = (0.5, 0.5)
BOUNDARY_PADDING = 0.95
BOUNDARY_DENY_PADDING = 0.01
CENTER_OBJECT_CONFIDENCE_THRESHOLD = 0.3
BOUNDINGBOX_WEIGHT = 0.5
# change param from 5 to 100
#KEEP_DATA_LENGTH = 5
KEEP_DATA_LENGTH = 100
CM_NODE_RATIO = 1.0
CM_MAX_VARIATION = 0.5
CM_CONFIDENCE = 0.5
# change param from 240 to 1000
RECALL_LENGTH = 1000
#RECALL_LENGTH = 240
RECALL_MATCH_THRESHOLD = 5
RECALL_CONFIDENCE = 0.3
BODY_PART_THRESHOLD = 0.3

REGISTER_MODEL_OVERLAP_RATIO=0.2
#ASSIGN_CONFIDENCE = 0.4
ASSIGN_CONFIDENCE = 0.3

class Track_own:
    def __init__(self, type, x1, y1, x2, y2, conf, img=None, featureType=''):
        self.id = -1
        self.type = type
        self.confidence = conf
        self.tl = (x1, y1)
        self.br = (x2, y2)
        self.rect = [x1, y1, x2 - x1, y2 - y1]
        self.dx = 0
        self.dy = 0
        self.match_count = 0
        self.tracking_count = 0
        self.last_state = 'Init'
        self.last_rect = self.rect
        self.last_frame = -1
        self.featureType = featureType

        self.feature = None


        if type == 'body':
            self.face = None
            self.upper = None
            self.bottom = None

    def match(self, new_trajectories):
        if len(new_trajectories) == 0:
            return None

        best_nn = None
        if self.type == 'body' :
            best_nn = self.match_body(new_trajectories)
        elif self.type == 'face':
            best_nn = self.match_face(new_trajectories)
        elif self.type == 'body(u)':
            best_nn = self.match_upper(new_trajectories)
        elif self.type == 'body(b)':
            best_nn = self.match_bottom(new_trajectories)

        return best_nn

    def calc_track_distance(self, new_track):
        dx = new_track.rect[0] - self.rect[0]
        dy = new_track.rect[1] - self.rect[1]
        dw = new_track.rect[2] - self.rect[2]
        dh = new_track.rect[3] - self.rect[3]
        geo_dist = math.sqrt((dx * dx) + (dy * dy) + ((dw * dw) + (dh * dh) / 2))

        if(self.featureType == 'vino_reid'):
            feature_dist = 0
            if self.feature is not None:
                feature_dist = get_distance_between_embeddings(self.feature, new_track.feature)

        return geo_dist + feature_dist

    def match_body(self, new_trajectories):
        best_body_dist = -1.0
        best_body_tr = None

        for n_tr in new_trajectories:

            # Find the nearest body
            if n_tr.id < 0 and n_tr.type == self.type :
                dx = n_tr.rect[0] - self.rect[0]
                dy = n_tr.rect[1] - self.rect[1]

                # Maximum move range per frame
                x_max = self.rect[2] * MAX_DELTA[0]
                y_max = self.rect[3] * MAX_DELTA[1]

                if -x_max <= dx <= x_max and -y_max <= dy <= y_max:
                    new_dist = self.calc_track_distance(n_tr)
                    if best_body_dist < 0 or best_body_dist > new_dist:
                        best_body_dist = new_dist
                        best_body_tr = n_tr

        return best_body_tr

    def match_face(self, new_trajectories):
        best_conf = 0.0
        best_tr = None

        for n_tr in new_trajectories:
            if n_tr.id < 0 and n_tr.type == 'face' and util.is_in_rect(n_tr.rect, self.rect):
                if n_tr.confidence > best_conf:
                    best_conf = n_tr.confidence
                    best_tr = n_tr

        return best_tr

    def match_upper(self, new_trajectories):
        best_conf = 0.0
        best_tr = None

        for n_tr in new_trajectories:
            if n_tr.id < 0 and n_tr.type == 'body(u)':
                overlap = util.getOverlapRatio(n_tr.rect, self.rect, is_rect=True)
                if overlap > BODY_PART_THRESHOLD and overlap > best_conf:
                    best_conf = overlap
                    best_tr = n_tr

        return best_tr

    def match_bottom(self, new_trajectories):
        best_conf = 0.0
        best_tr = None

        for n_tr in new_trajectories:
            if n_tr.id < 0 and n_tr.type == 'body(b)':
                overlap = util.getOverlapRatio(n_tr.rect, self.rect, is_rect=True)
                if overlap > BODY_PART_THRESHOLD and  overlap > best_conf:
                    best_conf = overlap
                    best_tr = n_tr

        return best_tr

    '''
    def find(self, img, last_img, candidate_features = (), candidate_tracks = (), feature_func=None):
        # Find by online-learner
        if self.use_cmatch and self.type == 'body' and self.classifier is not None and len(candidate_features) > 0:
            data = np.array(candidate_features)
            resp = self.classifier.predict(data)

            min_dist = 0
            best_candidate = None

            for ri in range(len(resp)):
                ct = candidate_tracks[ri]
                if resp[ri][1] > CM_CONFIDENCE:
                    dist = self.calc_track_distance(ct)
                    if best_candidate is None or dist < min_dist:
                        min_dist = dist
                        best_candidate = ct

            if best_candidate is not None:
                max_delta = self.last_rect[2] * CM_MAX_VARIATION
                min_x = self.last_rect[0] - max_delta
                max_x = self.last_rect[0] + max_delta
                min_y = self.last_rect[1] - max_delta
                max_y = self.last_rect[1] + max_delta
                min_w = self.last_rect[2] - max_delta
                max_w = self.last_rect[2] + max_delta
                min_h = self.last_rect[3] - max_delta
                max_h = self.last_rect[3] + max_delta


                if '' == 'D':
                    disp = img.copy()
                    cv2.rectangle(disp, best_candidate.tl, best_candidate.br, (0, 255, 255), 1)
                    cv2.imshow('CMatch', disp)
                    cv2.waitKey()

                if min_x <= best_candidate.tl[0] <= max_x and min_y <= best_candidate.tl[1] <= max_y and \
                    min_w <= best_candidate.rect[2] <= max_w and min_h <= best_candidate.rect[3] <= max_h:
                    self.last_state = 'CMatch'
                    self.tracker = None
                    return best_candidate

        # Find by tracker
        if self.tracker_type == '':
            return None

        if self.tracker is None:
            self.last_rect[0] = self.last_rect[0] if self.last_rect[0] >= 0 else 0
            self.last_rect[1] = self.last_rect[1] if self.last_rect[1] >= 0 else 0

            if self.tracker_type == 'KCF':
                self.tracker = KCFTracker.Tracker(last_img, self.last_rect)
            #elif self.tracker_type == 'TLD':
            #    self.tracker = TLDTracker.Tracker(last_img, self.last_rect)
            elif self.tracker_type == 'MF':
                self.tracker = MFTracker.Tracker(last_img, self.last_rect)
            elif self.tracker_type == 'TM':
                self.tracker = TMTracker.Tracker(last_img, self.last_rect, image_type='HistEQ',
                                                 max_delta_x=MAX_DELTA[0], max_delta_y=MAX_DELTA[1],
                                                 move_delta_x=MOVE_DELTA[0], move_delta_y=MOVE_DELTA[1])
            elif self.tracker_type == 'KLT':
                self.tracker = KLTTracker.Tracker(last_img, self.last_rect)
            else:
                raise Exception('Invalid self.tracker_type = ' + str(self.tracker_type))

            self.tracker.target_type = self.type
            self.last_state = 'Search'
            self.tracking_count = 0

        new_roi = self.tracker.update(img)
        n_tr = None
        if new_roi is not None :
            if self.classifier is not None and feature_func is not None:
                #f = feature_func(new_roi)
                f = feature_func(img, [new_roi])

                if is_ELM:
                    data = np.array(f)
                    test_data = elm.DataInput.DataSet(data, None)
                    resp = self.classifier.predict(test_data.images)

                
                if resp[0][1] < 0.5:
                    return None
            n_tr = Track('[Update]', new_roi[0], new_roi[1], new_roi[0] + new_roi[2], new_roi[1] + new_roi[3], self.tracker.confidence, None, featureType=self.featureType)
            if self.last_state == 'Search':
                self.tracking_count += 1
                if self.tracking_count > self.tracker_limit:
                    return None

        return n_tr
    '''
    def update(self, newTr, img, new_weight=1.0, candidate_features = ()):
        self.last_rect = self.rect

        if new_weight == 1.0:
            self.dx = newTr.tl[0] - self.tl[0]
            self.dy = newTr.tl[1] - self.tl[1]
            self.tl = newTr.tl
            self.br = newTr.br
            self.rect = newTr.rect
        else:
            ct1 = (int(self.rect[0] + (self.rect[2]/2)), int(self.rect[1] + (self.rect[3]/2)))
            ct2 = (int(newTr.rect[0] + (newTr.rect[2]/2)), int(newTr.rect[1] + (newTr.rect[3]/2)))

            self.dx = ct2[0] - ct1[0]
            self.dy = ct2[1] - ct2[0]

            w = int(((1 - new_weight) * self.rect[2]) + (new_weight * newTr.rect[2]))
            h = int(((1 - new_weight) * self.rect[3]) + (new_weight * newTr.rect[3]))
            x = int(ct2[0] - w / 2)
            y = int(ct2[1] - h / 2)

            if x < 0 :
                x = 0
            if y < 0 :
                y = 0
            if x + w > img.shape[1]:
                w = img.shape[1] - x
            if y + h > img.shape[0] :
                h = img.shape[0] - y

            self.tl = (x, y)
            self.br = (x + w, y + h)
            self.rect = [x, y, w, h]

        self.confidence = newTr.confidence

    def update_parts(self, new_trajectories, img):
        if self.type == 'body':
            def update_part(part):
                part.last_rect = part.rect
                new_part = part.match(new_trajectories)
                if new_part is not None:
                    part.update(new_part, img)
                    part.last_state = 'Match'
                    return part
                '''
                else :
                    new_part = part.find(img)
                    if new_part is not None and util.is_in_rect(new_part.rect, self.rect):
                        part.update(new_part, img)
                        part.last_state = 'Search'
                        return part
                    else:
                        return None
                '''
                return None

            if self.face is None:
                self.face = self.match_face(new_trajectories)
            else:
                self.face = update_part(self.face)

            if self.upper is None:
                self.upper = self.match_upper(new_trajectories)
            else:
                self.upper = update_part(self.upper)

            if self.bottom is None:
                self.bottom = self.match_bottom(new_trajectories)
            else:
                self.bottom = update_part(self.bottom)

    def copy(self):
        t = Track_own(self.type, self.tl[0], self.tl[1], self.br[0], self.br[1], self.confidence, featureType=self.featureType)
        t.dx = self.dx
        t.dy = self.dy
        t.last_state = self.last_state

        if self.type == 'body':
            if self.face is not None:
                t.face = self.face.copy()
            else:
                t.face = None
        return t

    @property
    def area(self):
        return self.rect[2] * self.rect[3]

class Tracking_own:
    def __init__(self, tracker_type='', tracker_limit=60, use_cmatch=False, featureType='', classifierType=''):
        self.UID = 0
        self.featureType = featureType
        self.classifierType = classifierType

        self.frameNo = 0
        self.candidate_features = None
        self.candidate_tracks = []
        self.lost_tracks = []
        self.last_image = None
        self.feature_func = None
        self.event_func = None
        self.profile_classifier = Profile.ProfileClassifier(self.classifierType)
    def find(self, img, last_image, new_list):
        pass


    def update(self, old_list, new_list, img):
        found_uid_list = []
        lost_target = []

        area_list = []
        for i in range(len(old_list)):
            area_list.append((i, old_list[i].area))
        sorted(area_list, reverse=True, key=lambda tr: tr[1])

        # Update existing one
        for tr_idx, tr_area in area_list:
            tr = old_list[tr_idx]

            n_tr = tr.match(new_list)

            # Lost, Search!
            if n_tr is None:
                # FIND the lost
                #n_tr = tr.find(img, last_img=self.LAST_IMAGE, feature_func=self.feature_func,candidate_features=self.candidate_features, candidate_tracks=self.candidate_tracks)
                print('k')

                # Search Succeed
                if n_tr is not None :
                    tr.update(n_tr, img, candidate_features=self.candidate_features)
                    tr.update_parts(new_list, img)
                    tr.last_frame = self.frameNo
                else:
                    lost_target.append(tr)
                    tr.last_state = 'Lost'

            # Found, Update!
            else:
                tr.update(n_tr, img, BOUNDINGBOX_WEIGHT, candidate_features=self.candidate_features)
                tr.update_parts(new_list, img)
                n_tr.id = tr.id
                found_uid_list.append(tr.id)
                tr.last_frame = self.frameNo
                tr.last_state = 'Match'
                tr.match_count += 1
                tr.tracker = None

                # Update Profile LHW
                self.profile_classifier.update_profile_with_uid(n_tr.id, n_tr.feature, img, n_tr.rect, n_tr.confidence)



        # Remove lost target from old list
        for tr in lost_target:
            old_list.remove(tr)
            
            # Add consistent track to lost tracks pool for later recall
            if tr.match_count > RECALL_MATCH_THRESHOLD:
                self.lost_tracks.append(tr)

        for idx, n_tr in enumerate(new_list):
            global UID
            if n_tr.id < 0 and n_tr.type == 'body':
                isNew = True
                isOverlap = False
                # Center prior - if it appears suddenly, the confidence should be high
                if n_tr.rect[0] > (img.shape[1] * (1 - BOUNDARY_PADDING)) or (n_tr.rect[0] + n_tr.rect[2]) > img.shape[1] or \
                    n_tr.rect[1] > (img.shape[0] * (1 - BOUNDARY_PADDING)) or (n_tr.rect[1] + n_tr.rect[3]) > img.shape[0]:
                    isNew = (n_tr.confidence > CENTER_OBJECT_CONFIDENCE_THRESHOLD)

                # Overlap prior - if it is included in or wrap around existing one, it is not valid
                if isNew:
                    wrap_around = False
                    wrap_idx = 0

                    for idx, tr in enumerate(old_list):
                        included = util.is_in_rect(n_tr.rect, tr.rect)
                        wrap_around = util.is_in_rect(tr.rect, n_tr.rect)
                        wrap_idx = idx

                        if included or wrap_around:
                            isNew = False
                            break

                    if wrap_around and n_tr.confidence > old_list[wrap_idx].confidence:
                        old_list[wrap_idx].update(n_tr, img, 1.0)
                if isNew:
                    # overlap with old_list
                    for idx, tr in enumerate(old_list):
                        overlap_ratio = util.getOverlapRatio(n_tr.rect, tr.rect, is_rect=True)
                        if overlap_ratio > REGISTER_MODEL_OVERLAP_RATIO:
                            isNew = False
                            break
                if isNew:
                    # Positioning on boundary is not allowed...
                    x1, y1, x2, y2 = n_tr.rect[0], n_tr.rect[1], n_tr.rect[0] + n_tr.rect[2], n_tr.rect[1] + n_tr.rect[3]

                    if x1 < (BOUNDARY_DENY_PADDING * img.shape[1]) or x2 > img.shape[1] * (1-BOUNDARY_DENY_PADDING):
                    #        or y1 < (BOUNDARY_DENY_PADDING * img.shape[0]) or  y2 > img.shape[0] * (1-BOUNDARY_DENY_PADDING):
                        isNew = False
                        break
                if isNew:
                    # Recall the lost tracks, should be recovered?
                    for tr in self.lost_tracks:
                        time_diff = self.frameNo - tr.last_frame
                        if tr.id >= 0 and tr.classifier is not None and time_diff <= RECALL_LENGTH:
                            f_shape = n_tr.feature.shape
                            fet = np.reshape(n_tr.feature, newshape=(1, f_shape[0]))
                            pred = tr.classifier.predict(fet)
                            if pred[0][1] > pred[0][0] and pred[0][1] > RECALL_CONFIDENCE:
                                n_tr.id = tr.id
                                n_tr.classifier = tr.classifier
                                tr.id = -1
                                break


                    if isOverlap == False:
                        # Recover with profile LHW
                        assign_uid_list = self.profile_classifier.classify_profile(n_tr.feature)
                        #assign_uid = self.profile_classifier.classify_profile_svm(n_tr.feature)
                        if len(assign_uid_list) == 0:
                            assign_uid = -1
                        else:
                            available_list = [0 for i in range(len(assign_uid_list))]
                            assign_uid = -1
                            for j in range(len(assign_uid_list)):
                                uid_exist = False
                                for i in range(len(found_uid_list)):
                                    if assign_uid_list[j][1] == found_uid_list[i]:
                                        uid_exist = True
                                if uid_exist == False:
                                    assign_uid = assign_uid_list[j][1]
                                    break

                        if n_tr.confidence >= ASSIGN_CONFIDENCE and assign_uid >= 0:
                            n_tr.id = assign_uid

                        if n_tr.id >= 0:
                            for tr in self.lost_tracks:
                                if tr.id == n_tr.id:
                                    n_tr.classifier = tr.classifier
                                    tr.id = -1
                                    break

                        # New track
                        #if n_tr.id < 0 and n_tr.confidence >= CM_CONFIDENCE:

                        if n_tr.id < 0:
                            n_tr.id = self.UID
                            found_uid_list.append(n_tr.id)
                            self.UID += 1
                            # register profile and update LHW
                            self.profile_classifier.register_profile(n_tr.id, n_tr.feature, img, n_tr.rect, n_tr.confidence, feature_type=self.featureType)
                        else:
                            isNew = False

                        """
                        img_cropped = img[n_tr.rect[1]:n_tr.rect[1] + n_tr.rect[3],
                                      n_tr.rect[0]:n_tr.rect[0] + n_tr.rect[2]]

                        for each_uid in range(self.UID):
                            each_profile = self.profile_classifier.get_profile_with_uid(each_uid)
                            # self.rect_image_list.append([current_image, current_rect])
                            curr_img, curr_rect = each_profile.rect_image_list[-1][0], each_profile.rect_image_list[-1][
                                1]
                            cv2.imshow(str(each_profile.uid), curr_img[curr_rect[1]:curr_rect[1] + curr_rect[3],
                                                              curr_rect[0]:curr_rect[0] + curr_rect[2]])

                        cv2.imshow('curr_img', img_cropped)
                        cv2.waitKey()
                        """



                        old_list.append(n_tr)
                """
                img_cropped = img[n_tr.rect[1]:n_tr.rect[1]+n_tr.rect[3], n_tr.rect[0]:n_tr.rect[0]+n_tr.rect[2]]
                cv2.imshow('test', img_cropped)
                cv2.waitKey()
                """

        # Forget the old lost tracks
        lost_target = []
        for tr in self.lost_tracks:
            time_diff = self.frameNo - tr.last_frame
            if tr.id >= 0:
                if time_diff <= RECALL_LENGTH:
                    lost_target.append(tr)

        self.lost_tracks = lost_target


        # Remove redundant tracks
        overlap_target = []
        for tr_1 in old_list:
            for tr_2 in old_list:
                if tr_1.id == tr_2.id:
                    continue

                if util.is_in_rect(tr_1.rect, tr_2.rect):
                    if tr_1.match_count > tr_2.match_count:
                        overlap_target.append(tr_2)
                        tr_2.id = -1
                    else:
                        overlap_target.append(tr_1)
                        tr_1.id = -1

                elif util.getOverlapRatio(tr_1.rect, tr_2.rect, is_rect=True) > 0.8:
                    if tr_1.confidence < tr_2.confidence:
                        overlap_target.append(tr_1)
                        #tr_2.id = tr_1.id
                    else:
                        overlap_target.append(tr_2)
                        #tr_1.id = tr_2.id


        for tr in overlap_target:
            try:
                old_list.remove(tr)
            except:
                pass

        self.LAST_IMAGE = img
        self.frameNo += 1