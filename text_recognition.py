# Implementing OpenCV OCR algorithm

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # looping over the number of columns
        for x in range(numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < 0.5:
                continue

            # compute offset factor as resulting feature maps
            # will be 4x smaller than the input image
            (offsetX, offsetY) = (x*4.0, y*4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and
            # height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute starting and ending (x, y) - coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of bounding boxes and associated confidences
    return (rects, confidences)

# laoding the input image and grab image dimensions
image = cv2.imread('img/example_01.jpg')
orig = image.copy()
(origH, origW) = image.shape[:2]

# set new width and height and determine ratio in change
# for both width and height
(newW, newH) = (320, 320)

rW = origW/float(newW)
rH = origH/float(newH)

# resize image and grab new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# defining the 2 output layer names for EAST detector model
layerNames = ['feature_fusion/Conv_7/Sigmoid',
                      'feature_fusion/concat_3']

# load the pre-trained EAST text detector model
print('[INFO] loading EAST text detector...')
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# construct a blob from the image and then perform a forward
# pass of the model to obtain the 2 output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W,H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

# decode the predictions, then apply non-maxima
# suppression to suppress weak, overlapping bounding boxes
(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

# initialize the list of results
results = []

# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
    # scale bounding box coordinates based on the respective ratios
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    # to obtain better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box
    padding = 0.0
    dX = int((endX - startX) * padding)
    dY = int((endY - startY) * padding)

    # apply padding to each side of the bounding box respectively
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX * 2))
    endY = min(origH, endY + (dY * 2))

    # extract the actual padded roi
    roi = orig[startY:endY, startX:endX]

    # to apply Tesseract v4 to OCR text we must supply
    # 1) language
    # 2) an OEM flag
    # 3) a PSM flag
    config = ('-l eng --oem 1 --psm 7')
    text = pytesseract.image_to_string(roi, config = config)

    results.append(((startX, startY, endX, endY), text))

# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r:r[0][1])

# loop over the results
for ((startX, startY, endX, endY), text) in results:
    print('OCR TEXT')
    print('===========')
    print(text)

    # strip out non-ASCII text so we can draw text on image using opencv
    # draw text and a bounding box surrounding text
    text = ''.join([c if ord(c) < 128 else '' for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # show the output image
    cv2.imshow('Text Detection: ', output)
    cv2.waitKey(0)
