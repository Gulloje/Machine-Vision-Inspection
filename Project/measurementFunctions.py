
import cv2
import numpy as np
def drawMeasurements( image, offsetBox, length, width):
    # Calculate midpoints for length and width text
    midPointLength = ((offsetBox[0][0] + offsetBox[1][0]) // 2 + 2, (offsetBox[0][1] + offsetBox[1][1]) // 2 + 2)
    midPointWidth = ((offsetBox[1][0] + offsetBox[2][0]) // 2 + 2, (offsetBox[1][1] + offsetBox[2][1]) // 2 + 2)

    # Draw text on the image

    cv2.putText(image, f"{length:.2f} mm", midPointLength, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1)
    cv2.putText(image, f"{width:.2f} mm", midPointWidth, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 1)

def displayMeasurements(imageToDisplay, actualMeasurement, dimension, yOffset):
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (10,50 + yOffset)
    color = getAppropriateColor(dimension, actualMeasurement)
    fontScale = .75
    thickness = 2

    text = f"Dimension: {dimension.name}  Actual: {actualMeasurement:.2f}"

    cv2.putText(imageToDisplay, text, origin, font, fontScale, color, thickness)

def drawCircleFromMinRect(imageToDisplay, contour, xOffset, yOffset):
    (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((x1, y1), (w1, h1),
                         rotation))
    box = np.intp(box)
    offsetBox = np.array([[pt[0] + xOffset, pt[1] + yOffset] for pt in box],
                         dtype=np.int32)

    center = (int(x1 + xOffset), int(y1 + yOffset))  # Center of the rect with offset
    diameter = max(w1, h1)  # The diameter is the max of width or height
    radius = int(diameter / 2)

    angle = 0
    point1X = int(center[0] + radius * np.cos(angle))
    point1Y = int(center[1] + radius * np.sin(angle))
    point1 = (point1X, point1Y)

    point2X = int(center[0] + radius * np.cos(angle + np.pi))  # Opposite side of the circle
    point2Y = int(center[1] + radius * np.sin(angle + np.pi))
    point2 = (point2X, point2Y)

    #cv2.drawContours(imageToDisplay, [offsetBox], 0, (0, 255, 0), 2)

    # Draw the circle using the calculated center and radius
    cv2.circle(imageToDisplay, center, radius, (0, 255, 0), 2)
    return point1, point2

def drawMinAreaRect(imageToDisplay, contour, xOffset, yOffset):
    (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((x1, y1), (w1, h1),
                         rotation))
    box = np.intp(box)
    offsetBox = np.array([[pt[0] + xOffset, pt[1] + yOffset] for pt in box],
                         dtype=np.int32)
    cv2.drawContours(imageToDisplay, [offsetBox], -1, (0, 255, 0), 2)
    return w1, h1, offsetBox

def drawCorrespondingDimension(dimension, imageToDisplay, contour, xOffset, yOffset):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .6
    thickness = 1
    if dimension.type == "Diameter":
        center, radius = drawCircleFromMinRect(imageToDisplay, contour, xOffset, yOffset)
        cv2.putText(imageToDisplay, str(dimension.name), (center[0], center[1] - 10), font, fontScale, (0, 255, 0), thickness)
        return center, radius
    elif dimension.type == "Height":
        w, h, offsetBox = drawMinAreaRect(imageToDisplay, contour, xOffset, yOffset)

        midpointX = int((offsetBox[1][0] + offsetBox[2][0]) / 2)
        midpointY = int((offsetBox[1][1] + offsetBox[2][1]) / 2)
        cv2.putText(imageToDisplay, str(dimension.name), (midpointX+5, midpointY),font, fontScale, (0, 255, 0), thickness)
        return offsetBox[1], offsetBox[2]
    elif dimension.type == "Width":
        w, h, offsetBox = drawMinAreaRect(imageToDisplay, contour, xOffset, yOffset)
        midpointX = int((offsetBox[0][0] + offsetBox[1][0]) / 2)
        midpointY = int((offsetBox[0][1] + offsetBox[1][1]) / 2)
        cv2.putText(imageToDisplay, str(dimension.name), (midpointX, midpointY-5), font, fontScale, (0, 255, 0), thickness)
        return offsetBox[0], offsetBox[1]

def getAppropriateColor(dimension, actualMeasurement):
    if float(dimension.min) <= float(actualMeasurement) <= float(dimension.max):
        return (0,255,0)
    else:
        return (0,0,255)