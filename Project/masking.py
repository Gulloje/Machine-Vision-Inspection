import sys, cv2, time, imutils
import numpy as np
from measurementFunctions import *



def getBinaryImage(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)
    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1) #generally just worse the more iterations, iteration 1 = thresh
    #cv2.imshow('Opening image', opening)

    cv2.imshow("Thresh", thresh)


    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # mess around with differe retr_... params, something about heirarchy

    contourLengths = []

    # Check if contours are found
    if len(contours) > 0:
        # Draw contours on the original image (or a copy of it)
        outline_image = image.copy()
        cv2.drawContours(outline_image, contours, -1, (0, 255, 0), 1)

        for contour in contours:
            contourLengths.append(cv2.contourArea(contour, False))

        # Display the binary thresholded image
        cv2.imshow("Thresh", thresh)

        return contours, contourLengths




def cropImageBorder(image, border_size=2):
    h, w = image.shape[:2]

    # Crop the image by removing 'border_size' pixels from each side
    cropped_image = image[border_size:h - border_size, border_size:w - border_size]

    return cropped_image

def processLiveFeed(image):
    h, w = image.shape[:2]

    if h != 720 and w != 1280:
        cropped = image
    else:
        cropped = image[:, 290:w -90]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 11)
    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image (adjusting offset for cropped section)
    #cv2.drawContours(image, contours, -1, (0, 0, 255), 1, offset=(290, 0))

    return image, opening, contours


def compareContours(contour1, contour2):
    hu1 = cv2.HuMoments(cv2.moments(contour1))
    hu2 = cv2.HuMoments(cv2.moments(contour2))
    #high value = not similar. #low value = more similar
    match = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
    print(match)
    return match



# for deleting and redrawing
def redrawContours(cleanImage, deletedContourLength, partDescriptor):
    image = cleanImage.copy()
    image, _, contours = processLiveFeed(image)
    #COMEBACK: if you delete 1 of length and width, the rectangle will not appear in the set dimensions mode
    for dimension in partDescriptor:
        for contour in contours:
            contourArcLen = cv2.arcLength(contour, True)
            print(f'Contour Length: {contourArcLen}, Deleted Length: {deletedContourLength}, Dimension Length: {dimension.contourArcLen}')
            if ((dimension.type == "Height" or dimension.type == "Width") and abs(contourArcLen - dimension.contourArcLen) < 15):
                drawDimension(dimension.type, image, contour)
            if abs(contourArcLen - deletedContourLength) > 15 and abs(contourArcLen - dimension.contourArcLen) < 15:
                drawDimension(dimension.type, image, contour)
                print('drawing dimension')
                break

    return image

def drawDimension(type, image, contour):
    if type == "Diameter":
        drawCircleFromMinRect(image, contour, 290, 0)
    elif type == "Height":
        drawMinAreaRect(image, contour, 290, 0, type)
    elif type == "Width":
        drawMinAreaRect(image, contour, 290, 0, type)


