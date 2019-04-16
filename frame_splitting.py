# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:18:22 2019

@author: Pronaya
"""

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

threshold = 4
url = "K:/pig_videos/Videos_ML_Project/"
fileName = "Train1_Cam3.mp4"
"""
sub = "MOG2"
if sub == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
"""    
    
capture = cv2.VideoCapture(url+fileName)
if not capture.isOpened:
    print('Unable to open: ' + url)
    exit(0)

frame_width = int(capture.get( cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
FPS = capture.get(cv2.CAP_PROP_FPS)
outLessMove = cv2.VideoWriter(r'K:\pig_videos\Videos_ML_Project\less_movement_'+fileName, cv2.VideoWriter_fourcc('A','V','C','1'), 
                              FPS, (frame_width,frame_height), True)
outHighMove = cv2.VideoWriter(r'K:\pig_videos\Videos_ML_Project\high_movement_'+fileName, cv2.VideoWriter_fourcc('A','V','C','1'), 
                              FPS, (frame_width,frame_height), True)

#capture.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
#length = capture.get(cv2.CAP_PROP_POS_MSEC)
totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
print("Tootal Frames: "+str(totalFrames) + "\n")

normList = []
while True:
    #capture.set(1, 3-1)    
    ret, frame = capture.read()
    if frame is None:
        break    
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    #frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    fnorm = np.linalg.norm(frame)
    normList.append(fnorm)
    #print(fnorm)
    """
    cv2.imshow('Frame', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    """
    #percent = (capture.get(cv2.CAP_PROP_POS_FRAMES)/totalFrames)*100
    if(int(capture.get(cv2.CAP_PROP_POS_FRAMES))%1000==0):
        percent = (capture.get(cv2.CAP_PROP_POS_FRAMES)/totalFrames)*100
        print(str(round(percent)) + "%")
        

plt.plot(normList[0:2000])

def smooth(x, box_pts, wh):
    if wh == "gaussian" :
        box = gaussian_filter(np.ones(box_pts), 1.5)
    elif wh == "box":
        box = np.ones(box_pts)/box_pts        
    x_smooth = np.convolve(x, box, mode='same')
    return x_smooth

# Smoothing the values using Savitzky–Golay filter ////https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# Any kinds of filters can be used
newNormList = savgol_filter(normList, 101, 3)
#newNormList = smooth(normList, 50, "gaussian")
plt.plot(newNormList[30:2000])



"""
newSubList = []
movingAvgList = []
movingavg = 0
for i in range(0, len(normList)):
    movingavg =movingavg + normList[i]
    if(i>0):
        movingavg = movingavg/2
    movingAvgList.append(movingavg)
    
    if((normList[i]-movingavg)>=0):
        newSubList.append(normList[i]-movingavg)


movingAvgList = movingAvgList - min(movingAvgList)
newSubList = newSubList -  min(newSubList)
plt.plot(newSubList)
plt.plot(movingAvgList[0:300])
"""



diffList = []
interval = int(FPS)*2
for i in range(interval, len(newNormList)):
    diff = 0
    for j in range(i-(interval-1), i):
        diff = diff + abs(newNormList[j]-newNormList[j-1])
        """
        if(j>i-(interval-1)):
            diff = diff/2
        """
    diff = diff/ len(range(i-(interval-1), i))
    diffList.append(diff)

"""
diffList = diffList - st.mean(diffList)   
diffList = list(diffList)
for i in range(len(diffList)):
    if (diffList[i] < 0):
        diffList[i] = 0
"""        
plt.plot(diffList[30:10000])    
    

   
 
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0)   
#capture.set(1, interval-1)  
cnt = 0
i = 0
selectedFramesSet = []
while True:
    ret, frame = capture.read()
    
    if(i>=len(diffList)):
        break
    
    if(diffList[i]>threshold):
        i = i+interval
        if(i>totalFrames):
            break
        capture.set(1, i-1)  
        continue
    
    cnt = cnt+1
    if frame is None:
        break
    
    #print(cnt)
    #print(ret)
    #fgMask = backSub.apply(frame)
    
    selectedFramesSet.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    outLessMove.write(frame)
    
    """
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
    cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgMask)
    
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    """
    
    i = i + 1 
    
    if(i%1000==0):
        percent = (i/totalFrames)*100
        print(str(round(percent)) + "%")

outLessMove.release()    
totaFramesSet = list(range(0, int(totalFrames)))    
movingFramesSet = list(set(totaFramesSet) - set(selectedFramesSet))
movingFramesSet.sort()
selectedFramesSet.sort()



# ovservation of movement
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0) 
j = 0
jMax = len(movingFramesSet)
for i in movingFramesSet:
    capture.set(1, i-1)  
    ret, frame = capture.read() 
    
    outHighMove.write(frame)
    
    """
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    cv2.imshow('Frame', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    """
    j = j + 1  
    if(j%100==0):
        percent = (j/jMax)*100
        print(str(round(percent)) + "%")

outHighMove.release()    




# ovservation of less 
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0) 
for i in selectedFramesSet:
    capture.set(1, i-1)  
    ret, frame = capture.read()
    
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
    cv2.imshow('Frame', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break



# Closes all the frames
cv2.destroyAllWindows()    