# extract color histogram and save it
from calcHistogram import calcHistogramHSV
from calcHistogram import calcHistogram3D
from calcHistogram import calcHistFeature

import os
import cv2
import numpy as np

import DataProvider

def extractColorHistogramParent(annotationParentPath, videoParentPath, frameParentPath, outParentPath, bins=8, HSV=False, Hist_3D=False):
    videoFolder = os.listdir(annotationParentPath)

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
                bbox = objInfoSplit[3:]

                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                objList.append([x1, y1, x2, y2])


            frameIdx = int(eachFramePath.split('.')[0])
            frameObjList.append((frameIdx , objList))

        frameObjList.sort(key=lambda element: element[0])

        # Load video info

        videoPath = os.path.join(videoParentPath, eachVideo + '.mp4')

        dp = DataProvider.VideoDataProvider(videoPath)

        frameIdx = 0
        objDetectidx = 0

        resizeRate = 0.4

        if(HSV == True):
            outFolderAbsPath = os.path.join(outParentPath, 'human_track_data_kaist-color_HSV')
        elif(Hist_3D == True):
            outFolderAbsPath = os.path.join(outParentPath, 'human_track_data_kaist-color_3D')
        else:
            outFolderAbsPath = os.path.join(outParentPath, 'human_track_data_kaist-color_' + str(bins))

        if (not os.path.exists(outFolderAbsPath)):
            os.mkdir(outFolderAbsPath)

        outFolderAbsPath = os.path.join(outFolderAbsPath, eachVideo)

        if (not os.path.exists(outFolderAbsPath)):
            os.mkdir(outFolderAbsPath)

        while(objDetectidx < len(frameObjList)):
            img = dp.get()
            if img is None:
                break

            img_resize = cv2.resize(img, (0, 0), fx=resizeRate, fy=resizeRate)



            if(frameObjList[objDetectidx][0] == frameIdx and objDetectidx < len(frameObjList)):
                bboxList = frameObjList[objDetectidx][1]
                imgDraw = img_resize

                objIdx = 0

                objDetectidx += 1

                for eachBbox in bboxList:
                    if(HSV == True):
                        featureArr = calcHistogramHSV(imgDraw, [eachBbox])
                    elif(Hist_3D == True):
                        featureArr = calcHistogram3D(imgDraw, [eachBbox])
                    else:
                        featureArr = calcHistFeature(imgDraw, [eachBbox], bins)

                    outAbsPath = os.path.join(outFolderAbsPath, str(frameIdx) + '_' + str(objIdx) + '.npy')

                    with open(outAbsPath, 'w') as writeFile:
                        np.save(writeFile, featureArr)

                    objIdx += 1

            frameIdx+=1




def main():
    annotationParentPath = '../data/human_track_data_kaist/annotation'
    frameParentPath = 'out/'

    videoParentPath = '../data/human_track_data_kaist/video'

    HSV = False
    Hist_3D = True

    colorBinsList = [8,16,32,64]


    if(HSV == True or Hist_3D == True):
        extractColorHistogramParent(annotationParentPath, videoParentPath, frameParentPath, 'feature/colorHist', 8,
                                    HSV, Hist_3D)
    else:
        for colorBins in colorBinsList:
            extractColorHistogramParent(annotationParentPath, videoParentPath, frameParentPath, 'feature/colorHist', colorBins, HSV, Hist_3D)

if(__name__ == '__main__'):
    main()





