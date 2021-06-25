# extract frame

import os
import cv2
import numpy as np
import shutil

import DataProvider

def extractVideoFromAnnoFile(video_parent_path, anno_parent_path, out_video_path):
    anno_path_list = os.listdir(anno_parent_path)

    for each_anno in anno_path_list:
        print('Processing for ' + each_anno)
        each_video_abs_path = os.path.join(video_parent_path,
                                           each_anno + '.mp4')
        if os.path.exists(each_video_abs_path):
            src_path = each_video_abs_path
            dst_path = os.path.join(out_video_path, each_anno + '.mp4')

            shutil.copy(src_path, dst_path)
        else:
            print('Not exist for ' + each_anno)




def extractwholeFrame(videoParentPath, outParentPath):

    videoFolder = os.listdir(videoParentPath)

    if not os.path.exists(outParentPath):
        os.mkdir(outParentPath)

    for eachVideo in videoFolder:
        print('Processing ' + eachVideo)

        videoPath = os.path.join(videoParentPath, eachVideo)

        dp = DataProvider.VideoDataProvider(videoPath)

        resizeRate = 0.4

        outFolderAbsPath = os.path.join(outParentPath, eachVideo)

        if (not os.path.exists(outFolderAbsPath)):
            os.mkdir(outFolderAbsPath)

        frameIdx = 0
        while(1):
            img = dp.get()
            if img is None:
                break
            img_resize = cv2.resize(img, (0, 0), fx=resizeRate, fy=resizeRate)

            outAbsPath = os.path.join(outFolderAbsPath, str(frameIdx) + '.png')

            frameIdx += 1
            cv2.imwrite(outAbsPath, img_resize)


def extractFrameParent(annotationParentPath, videoParentPath, outParentPath):
    videoFolder = os.listdir(annotationParentPath)

    if not os.path.exists(outParentPath):
        os.mkdir(outParentPath)

    for eachVideo in videoFolder:
        print('Processing ' + eachVideo)
        eachVideoAbsPath = os.path.join(annotationParentPath, eachVideo)

        framePathList = os.listdir(eachVideoAbsPath)

        frameObjList = []

        for eachFramePath in framePathList:

            eachFrameAbsPath = os.path.join(eachVideoAbsPath, eachFramePath)
            print('Processing ' + eachFrameAbsPath)

            # Load bbox info
            with open(eachFrameAbsPath) as readFile:
                allLines = readFile.readlines()

            objectNum = int(allLines[0])

            objList = []

            for i in range(objectNum):
                objInfoSplit = allLines[i+1].split(',')
                objIdx = int(objInfoSplit[1])
                bbox = objInfoSplit[3:]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                objList.append([objIdx, x1, y1, x2, y2])


            frameIdx = int(eachFramePath.split('.')[0])
            frameObjList.append((frameIdx , objList))

        frameObjList.sort(key=lambda element: element[0])

        # Load video info

        videoPath = os.path.join(videoParentPath, eachVideo + '.mp4')

        dp = DataProvider.VideoDataProvider(videoPath)

        frameIdx = 0
        objDetectidx = 0

        resizeRate = 0.4

        outFolderAbsPath = os.path.join(outParentPath, eachVideo)

        if (not os.path.exists(outFolderAbsPath)):
            os.mkdir(outFolderAbsPath)

        while(objDetectidx < len(frameObjList)):
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resizeRate, fy=resizeRate)

            if(frameObjList[objDetectidx][0] == frameIdx and objDetectidx < len(frameObjList)):
                bboxList = frameObjList[objDetectidx][1]


                objDetectidx += 1



                for eachBbox in bboxList:
                    objIdx = eachBbox[0]

                    outAbsPath = os.path.join(outFolderAbsPath, str(frameIdx) + '_' + str(objIdx) + '.png')

                    imgDraw = img_resize[eachBbox[2]:eachBbox[4], eachBbox[1]:eachBbox[3]]

                    #cv2.imshow('test', imgDraw)
                    #cv2.waitKey()

                    cv2.imwrite(outAbsPath, imgDraw)

            frameIdx+=1





'''
# extract human frame
#annotationParentPath = '../data/human_track_data_kaist/annotation'
#frameOutparent = '../data/human_track_data_kaist/frame'
#videoParentPath = '../data/human_track_data_kaist/video'

#annotationParentPath = '../data/human_track_data_kaist_certification/annotation'
#frameOutparent = '../data/human_track_data_kaist_certification/frame'
#videoParentPath = '../data/human_track_data_kaist_certification/video'

annotationParentPath = '../data/human_track_data_kaist_realenv/annotation'
frameOutparent = '../data/human_track_data_kaist_realenv/frame'
videoParentPath = '../data/human_track_data_kaist_realenv/video'

extractFrameParent(annotationParentPath, videoParentPath, frameOutparent)
'''

'''
# extract all frame

#frameOutparent = '../data/human_track_data_kaist/wholeFrame'
#videoParentPath = '../data/human_track_data_kaist/video'

#frameOutParent = '../data/human_track_data_kaist_certification/wholeFrame'
#videoParentPath = '../data/human_track_data_kaist_certification/video'

frameOutParent = '../data/human_track_data_kaist_realenv/wholeFrame'
videoParentPath = '../data/human_track_data_kaist_realenv/video'

extractwholeFrame(videoParentPath, frameOutParent)
'''

video_parent_path = '../data/human_track_data_kaist/video'
anno_parent_path = '../data/human_track_data_kaist_certification/annotation'
out_video_path = '../data/human_track_data_kaist_certification/video'

extractVideoFromAnnoFile(video_parent_path, anno_parent_path, out_video_path)



