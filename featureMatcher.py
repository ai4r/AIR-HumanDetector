# feature matcher
import os
import numpy as np

from calcHistogram import compareHistFeature
from calcHistogram import compareHistFeatureHSV
from calcHistogram import compareHistFeature3D

# load feature
def loadFeature(featurePath):

    featureList = []

    frameParentPath = os.listdir(featurePath)

    for eachFrame in frameParentPath:
        eachFrameAbsPath = os.path.join(featurePath, eachFrame)


        frameIdx = int(eachFrame.split('_')[0])
        objidx = int(eachFrame.split('_')[1].split('.')[0])

        with open(eachFrameAbsPath) as readFile:
            featureArr = np.squeeze(np.load(readFile))

        featureList.append((frameIdx, objidx, featureArr, eachFrameAbsPath))



    featureList.sort(key=lambda element: element[0])

    return featureList

def featureComparisonInnerVideo(featureList, currFeatureCompare=10, metric = 'euclidean'):

    featureID_1_List = []
    featureID_2_List = []

    # set the reference feature set
    for eachFrame in featureList:
        frameIdx, objIdx, featureArr = eachFrame[0], eachFrame[1], eachFrame[2]

        if(objIdx == 0 and len(featureID_1_List) == 0):
            featureID_1_List.append((frameIdx, objIdx, featureArr))
        if(objIdx == 1 and len(featureID_2_List) == 0):
            featureID_2_List.append((frameIdx, objIdx, featureArr))

        if(len(featureID_1_List) > 0 and len(featureID_2_List) > 0):
            break

    for eachFrame in featureList:
        frameIdx, objIdx, featureArr = eachFrame[0], eachFrame[1], eachFrame[2]

        # Compare feature 1 list
        avg_ID_1_Dist = 0
        cnt_ID_1 = 0
        for i in range(currFeatureCompare):
            if(i >= len(featureID_1_List)):
                break
            if(metric == 'HSV'):
                dist = compareHistFeatureHSV(featureArr, featureID_1_List[i][2])
            elif(metric == '3D'):
                dist = compareHistFeature3D(featureArr, featureID_1_List[i][2])
            else:
                dist = compareHistFeature(featureArr, featureID_1_List[i][2], metricType=metric)
            avg_ID_1_Dist += dist
            cnt_ID_1 += 1

        avg_ID_1_Dist = avg_ID_1_Dist / cnt_ID_1

        avg_ID_2_Dist = 0

        cnt_ID_2 = 0

        # Compare feature 2 list
        for i in range(currFeatureCompare):
            idx_curr = -i
            if(i >= len(featureID_2_List)):
                break

            if(metric == 'HSV'):
                dist = compareHistFeatureHSV(featureArr, featureID_2_List[i][2])
            elif (metric == '3D'):
                dist = compareHistFeature3D(featureArr, featureID_2_List[i][2])
            else:
                dist = compareHistFeature(featureArr, featureID_2_List[i][2], metricType=metric)
            avg_ID_2_Dist += dist
            cnt_ID_2 += 1

        avg_ID_2_Dist = avg_ID_2_Dist / cnt_ID_2

        if(avg_ID_1_Dist <= avg_ID_2_Dist):
            featureID_1_List.append((frameIdx, objIdx, featureArr))
        else:
            featureID_2_List.append((frameIdx, objIdx, featureArr))

    # Check feature

    cnt_1_correct = 0
    cnt_2_correct = 0

    for eachFeature in featureID_1_List:
        if(eachFeature[1] == 0):
            cnt_1_correct+=1
    for eachFeature in featureID_2_List:
        if(eachFeature[1] == 1):
            cnt_2_correct+=1

    ID_1_acc = float(cnt_1_correct) / len(featureID_1_List)
    ID_2_acc = float(cnt_2_correct) / len(featureID_2_List)
    tot_acc = float(cnt_1_correct + cnt_2_correct) / (len(featureID_1_List) + len(featureID_2_List))

    return ID_1_acc, ID_2_acc, tot_acc











def featureComparisonOuterVideo():
    pass

def featureMatcher():
    pass


def main():
    colorBinList = [8,16,32,64]
    metricTypeList = ['euclidean', 'correlation']
    HSV = False
    hist_3D = True

    if(HSV):
        pathOrigin = 'feature/colorHist/human_track_data_kaist-color_HSV'
        pathFrame = '../'

        outPath = 'out_HSV.csv'

        folderPath = os.listdir(pathOrigin)

        resList = []

        for eachFolder in folderPath:
            print('Processing for ' + eachFolder)
            featureList = loadFeature(os.path.join(pathOrigin, eachFolder))
            ID_1_acc, ID_2_acc, tot_acc = featureComparisonInnerVideo(featureList, currFeatureCompare=10,
                                                                      metric="HSV")

            resList.append((eachFolder, ID_1_acc, ID_2_acc, tot_acc))

        with open(outPath, 'w') as writeFile:
            writeFile.write('VideoName, ID1_acc, ID2_acc, Total_acc \n ')
            for eachRes in resList:
                writeFile.write('%s, %f, %f, %f \n ' % (eachRes[0], eachRes[1], eachRes[2], eachRes[3]))
    elif(hist_3D == True):
        pathOrigin = 'feature/colorHist/human_track_data_kaist-color_3D'

        outPath = 'out_3D.csv'

        folderPath = os.listdir(pathOrigin)

        resList = []

        for eachFolder in folderPath:
            print('Processing for ' + eachFolder)
            featureList = loadFeature(os.path.join(pathOrigin, eachFolder))
            ID_1_acc, ID_2_acc, tot_acc = featureComparisonInnerVideo(featureList, currFeatureCompare=10,
                                                                      metric='3D')

            resList.append((eachFolder, ID_1_acc, ID_2_acc, tot_acc))

        with open(outPath, 'w') as writeFile:
            writeFile.write('VideoName, ID1_acc, ID2_acc, Total_acc \n ')
            for eachRes in resList:
                writeFile.write('%s, %f, %f, %f \n ' % (eachRes[0], eachRes[1], eachRes[2], eachRes[3]))
    else:
        for colorBin in colorBinList:
            for metricType in metricTypeList:
                pathOrigin = 'feature/colorHist/human_track_data_kaist-color_' + str(colorBin)

                outPath = 'out_' + metricType + '_' + str(colorBin) + '.csv'

                if(os.path.exists(outPath)):
                    continue

                folderPath = os.listdir(pathOrigin)

                resList = []

                for eachFolder in folderPath:
                    print('Processing for ' + eachFolder)
                    featureList = loadFeature(os.path.join(pathOrigin, eachFolder))
                    ID_1_acc, ID_2_acc, tot_acc = featureComparisonInnerVideo(featureList, currFeatureCompare=10, metric=metricType)

                    resList.append((eachFolder, ID_1_acc, ID_2_acc, tot_acc))

                with open(outPath, 'w') as writeFile:
                    writeFile.write('VideoName, ID1_acc, ID2_acc, Total_acc \n ')
                    for eachRes in resList:
                        writeFile.write('%s, %f, %f, %f \n ' % (eachRes[0], eachRes[1], eachRes[2], eachRes[3]))





if __name__ == '__main__':
    main()




