'''
OPENVINO Reidentification module
'''

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'openvino'))
sys.path.insert(0, os.path.join(os.getcwd(), 'openvino', 'openvino_lib'))

from utils.network_wrappers import VectorCNN
from openvino_lib.openvino.inference_engine import IECore
import cv2
from scipy.spatial.distance import cosine
import numpy as np
from calcHistogram import calcCropped



class ReidMod:
    def __init__(self, model_path):
        self.load_network = False
        self.initialize_network(model_path)

    def initialize_network(self, model_path):
        if not self.load_network:
            ie = IECore()
            # FOR FASTER CPU CALCULATION
            ie.set_config(config={"CPU_BIND_THREAD": "NO"}, device_name="CPU")
            self.reidentification_net = VectorCNN(ie, model_path)



            self.load_network = True
        else:
            print('Network is already initialized')

    def get_embedding(self, image):
        self.embedding_vector = self.reidentification_net.forward([image])
        return self.embedding_vector

#reid_model_path = os.path.join(os.getcwd(),'openvino_lib', 'model/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml')
if __name__ == '__main__':
    reid_model_path = os.path.join(os.getcwd(),'model/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml')
else:
    reid_model_path = os.path.join(os.getcwd(),'openvino','model/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml')
REIDMOD = ReidMod(reid_model_path)

def get_reid_feature(img, rectList, cropping_ratio=(1.0,1.0)):
    featureList = []
    for eachRect in rectList:
        imgRect, x1, x2, y1, y2 = calcCropped(img, eachRect,)
        embedding = REIDMOD.get_embedding(imgRect)
        featureList.append(embedding)

    featureArr = np.asarray(featureList)

    return featureArr

def get_distance_between_embeddings(feature1, feature2):
    dist = cosine(feature1, feature2)
    return dist


def test_reid_embedding():
    reid_model_path = 'model/person-reidentification-retail-0288/FP32/person-reidentification-retail-0288.xml'

    #test_imgA_1_path = '../testSample/out/10_0.png'
    #test_imgA_2_path = '../testSample/out/24_0.png'
    #test_imgB_1_path = '../testSample/out/221_1.png'
    #test_imgB_2_path = '../testSample/out/492_1.png'

    #test_imgA_1_path = '../testSample/res_frame/0_0.png'
    #test_imgA_2_path = '../testSample/res_frame/1_0.png'

    test_imgA_1_path = "../210602_result/human_track_real_data_kaist_vino_reid/test_a'-04-01-3p/49_0.png"
    test_imgA_2_path = "../210602_result/human_track_real_data_kaist_vino_reid/test_a'-04-01-3p/50_1.png"

    test_imgB_1_path = '../testSample/res_frame/0_1.png'
    test_imgB_2_path = '../testSample/res_frame/1_1.png'


    reid = ReidMod(reid_model_path)

    test_imgA_1 = cv2.imread(test_imgA_1_path)
    test_imgA_2 = cv2.imread(test_imgA_2_path)
    test_imgB_1 = cv2.imread(test_imgB_1_path)
    test_imgB_2 = cv2.imread(test_imgB_2_path)

    featureA_1 = reid.get_embedding(test_imgA_1)
    featureA_2 = reid.get_embedding(test_imgA_2)
    featureB_1 = reid.get_embedding(test_imgB_1)
    featureB_2 = reid.get_embedding(test_imgB_2)

    dist_intra_A = cosine(featureA_1, featureA_2)
    dist_intra_B = cosine(featureB_1, featureB_2)

    dist_inter_AB11 = cosine(featureA_1, featureB_1)
    dist_inter_AB12 = cosine(featureA_1, featureB_2)

    dist_inter_AB21 = cosine(featureA_2, featureB_1)
    dist_inter_AB22 = cosine(featureA_2, featureB_2)



    print('dist_intra_A: {}\ndist_intra_B: {}\ndist_inter_AB11: {}\ndist_inter_AB12:{}\ndist_inter_AB21: {}\ndist_inter_AB22: {}\n'.format(dist_intra_A, dist_intra_B, dist_inter_AB11, dist_inter_AB12, dist_inter_AB21, dist_inter_AB22))

    print('THE Same feature: {}'.format(cosine(featureA_1, featureA_1)))

    cv2.imshow('testA_1', test_imgA_1)
    cv2.imshow('testA_2', test_imgA_2)
    cv2.waitKey()





def main():
    test_reid_embedding()



if __name__ == '__main__':
    main()
