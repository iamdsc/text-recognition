# Text Detection in natural scene images using EAST text detector
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2

# load input image and grab image dimensions
image=cv2.imread('img/lebron_james.jpg')
orig=image.copy()
(H,W)=image.shape[:2]

(newW,newH)=(320,320)
rW=W/newW
rH=H/newH

# resize image and grab new image dimensions
image=cv2.resize(image,(newW,newH))
(H,W)=image.shape[:2]

# define output layer names for the EAST detector model
layerNames=['feature_fusion/Conv_7/Sigmoid','feature_fusion/concat_3']

# load the pre-trained EAST text detector
print('[INFO] loading EAST text detector')
net=cv2.dnn.readNet('frozen_east_text_detection.pb')

# construct a blob from image
blob=cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68,116.78,103.94),swapRB=True,crop=False)

# perform a forward pass of the model to obtain output layer sets
net.setInput(blob)
(scores,geometry)=net.forward(layerNames)

(numRows,numCols)=scores.shape[2:4]
rects=[] # stores the bounding box (x,y)-coordinates for text regions
confidences=[]  # stores the prob associated with each of the bounding boxes

# loop over the number of rows
for y in range(numRows):
    # extract the scores and geometrical data
    scoresData=scores[0,0,y]
    xData0=geometry[0,0,y]
    xData1=geometry[0,1,y]
    xData2=geometry[0,2,y]
    xData3=geometry[0,3,y]
    anglesData=geometry[0,4,y]

    # loop over the number of cols
    for x in range(numCols):
        # ignore score with insufficient prob
        if scoresData[x] < 0.5:
            continue
        # compute offset factor
        (offsetX,offsetY)=(x*4.0,y*4.0)
        # extract rotational angle and compute sine and cosine
        angle=anglesData[x]
        cos=np.cos(angle)
        sin=np.sin(angle)

        # derive width and height of bounding box
        h=xData0[x]+xData2[x]
        w=xData1[x]+xData3[x]

        # compute starting and ending (x,y)-coordinates
        endX=int(offsetX+(cos*xData1[x])+(sin*xData2[x]))
        endY=int(offsetY-(sin*xData1[x])+(cos*xData2[x]))
        startX=int(endX-w)
        startY=int(endY-h)

        rects.append((startX,startY,endX,endY))
        confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak overlapping bounding boxes
boxes=non_max_suppression(np.array(rects),probs=confidences)

# loop over the bounding boxes
for (startX,startY,endX,endY) in boxes:
    # scale the bounding box coordinates based on the respective ratios
    startX=int(startX*rW)
    startY=int(startY*rH)
    endX=int(endX*rW)
    endY=int(endY*rH)

    # draw the bounding box on the image
    cv2.rectangle(orig,(startX,startY),(endX,endY),(0,255,0),2)

# show the output image
cv2.imshow('Text Detection',orig)
cv2.waitKey(0)
