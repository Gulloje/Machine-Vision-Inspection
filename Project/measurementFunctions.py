
import cv2
import numpy as np

import math

def displayMeasurements(imageToDisplay, actualMeasurement, dimension, yOffset):
    font = cv2.FONT_HERSHEY_SIMPLEX
    origin = (10,50 + yOffset)
    color = getAppropriateColor(dimension, actualMeasurement)
    fontScale = .75
    thickness = 2

    text = f"Dimension: {dimension.name}  Actual: {actualMeasurement:.3f}"

    cv2.putText(imageToDisplay, text, origin, font, fontScale, color, thickness)
def displayMeasurementName(image, pt1, pt2, dimensionName):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (215,215,215)
    
    fontScale = .7
    thickness = 1

    midX = int((pt1[0] + pt2[0]) /2)
    midY = int((pt1[1] + pt2[1]) /2)

    cv2.putText(image, str(dimensionName), (midX + 5, midY), font, fontScale, (0, 0, 0), 2)
    cv2.putText(image, str(dimensionName), (midX + 5, midY), font, fontScale, color, thickness)


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




def drawMinAreaRect(imageToDisplay, contour, xOffset, yOffset, type):
    (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((x1, y1), (w1, h1),
                         rotation))
    box = np.intp(box)
    offsetBox = np.array([[pt[0] + xOffset, pt[1] + yOffset] for pt in box],
                         dtype=np.int32)
    print(rotation)
    #cv2.drawContours(imageToDisplay, [offsetBox], -1, (0, 255, 0), 2)


    if -45 <= rotation < 45:

        if type == 'Width':
            left = [offsetBox[1], offsetBox[2]]
            right = [offsetBox[0], offsetBox[3]]
            cv2.line(imageToDisplay, tuple(left[0]), tuple(left[1]), (0, 255, 0), 2)
            cv2.line(imageToDisplay, tuple(right[0]), tuple(right[1]), (0, 255, 0), 2)
            return left[0], left[1]
        else:
            # Draw top and bottom for height
            top = [offsetBox[0], offsetBox[1]]
            bottom = [offsetBox[2], offsetBox[3]]
            cv2.line(imageToDisplay, tuple(top[0]), tuple(top[1]), (0, 255, 0), 2)
            cv2.line(imageToDisplay, tuple(bottom[0]), tuple(bottom[1]), (0, 255, 0), 2)
            return top[0], top[1]
    else:
        # Assume vertical alignment; width is along [1, 2] and [0, 3]
        if type == 'Width':
            top = [offsetBox[0], offsetBox[1]]
            bottom = [offsetBox[2], offsetBox[3]]
            cv2.line(imageToDisplay, tuple(top[0]), tuple(top[1]), (0, 255, 0), 2)
            cv2.line(imageToDisplay, tuple(bottom[0]), tuple(bottom[1]), (0, 255, 0), 2)
            return top[0], top[1]
        else:
            # Draw left and right for height
            left = [offsetBox[1], offsetBox[2]]
            right = [offsetBox[0], offsetBox[3]]
            cv2.line(imageToDisplay, tuple(left[0]), tuple(left[1]), (0, 255, 0), 2)
            cv2.line(imageToDisplay, tuple(right[0]), tuple(right[1]), (0, 255, 0), 2)
            return left[0], left[1]

def drawClosestSide(imageToDisplay, contour, xOffset, yOffset, dimension):
    (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
    box = cv2.boxPoints(((x1, y1), (w1, h1),
                         rotation))
    box = np.intp(box)
    offsetBox = np.array([[pt[0] + xOffset, pt[1] + yOffset] for pt in box],
                         dtype=np.int32)

    sides = [
        (offsetBox[0], offsetBox[1]),
        (offsetBox[1], offsetBox[2]),
        (offsetBox[2], offsetBox[3]),
        (offsetBox[3], offsetBox[0])
    ]

    closestWidthSide, closestWidthOp, closestHeightSide, closestHeightOp = dimension.findClosestSide(sides)
    yName, xName = findDimensionNamingSides(closestWidthSide, closestWidthOp, closestHeightSide, closestHeightOp)

    if dimension.type == "Width" and closestWidthSide:
        cv2.line(imageToDisplay, tuple(closestWidthSide[0]), tuple(closestWidthSide[1]), (0, 255, 0), 2)
        cv2.line(imageToDisplay, tuple(closestWidthOp[0]), tuple(closestWidthOp[1]), (0, 255, 0), 2)
        displayMeasurementName(imageToDisplay, yName[0], yName[1], dimension.name)
        return closestWidthSide[0], closestWidthSide[1]
    elif dimension.type == "Height" and closestHeightSide:
        cv2.line(imageToDisplay, tuple(closestHeightSide[0]), tuple(closestHeightSide[1]), (0, 255, 0), 2)
        cv2.line(imageToDisplay, tuple(closestHeightOp[0]), tuple(closestHeightOp[1]), (0, 255, 0), 2)
        displayMeasurementName(imageToDisplay, xName[0], xName[1], dimension.name)
        return closestHeightSide[0], closestHeightSide[1]




def drawCorrespondingDimension(dimension, imageToDisplay, contour, xOffset, yOffset):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .6
    thickness = 1
    if dimension.type == "Diameter":
        center, radius = drawCircleFromMinRect(imageToDisplay, contour, xOffset, yOffset)
        cv2.putText(imageToDisplay, str(dimension.name), (center[0], center[1] - 10), font, fontScale, (0, 0, 0), 2)
        cv2.putText(imageToDisplay, str(dimension.name), (center[0], center[1] - 10), font, fontScale, (255, 255, 255), thickness)
        return center, radius
    elif dimension.type == "Height":
        pt1, pt2 = drawClosestSide(imageToDisplay, contour, xOffset, yOffset, dimension)
        return pt1, pt2
    elif dimension.type == "Width":
        pt1, pt2 = drawClosestSide(imageToDisplay, contour, xOffset, yOffset, dimension)
        return pt1, pt2
    elif dimension.type == "Radius":

        pt1, pt2 = findKnownCircles(imageToDisplay, contour, dimension)
        cv2.putText(imageToDisplay, str(dimension.name), (pt1[0], pt1[1] - 10), font, fontScale, (0, 0, 0),
                    2)
        cv2.putText(imageToDisplay, str(dimension.name), (pt1[0], pt1[1] - 10), font, fontScale, (255, 255, 255),
                    thickness)
        # for i, _ in enumerate(allCenters):
        #     cv2.putText(imageToDisplay, str(dimension.name), (allCenters[i][0]+5, allCenters[i][1]+5), font, fontScale, (0, 255, 0), thickness)


        return pt1, pt2




def getAppropriateColor(dimension, actualMeasurement):
    if float(dimension.min) <= float(actualMeasurement) <= float(dimension.max):
        return (0,255,0)
    else:
        return (0,0,255)

def findDimensionNamingSides(closestWidthSide, closestWidthOpposite, closestHeightSide, closestHeightOpposite):

    maxY = float('inf')
    maxYWidthSide = None


    for side in [closestWidthSide, closestWidthOpposite]:
        if side:
            sideMaxY = min(side[0][1], side[1][1])
            if sideMaxY < maxY:
                maxY = sideMaxY
                maxYWidthSide = side


    maxX = float('-inf')
    maxXHeightSide = None


    for side in [closestHeightSide, closestHeightOpposite]:
        if side:
            sideMaxX = max(side[0][0], side[1][0])  # Compare x-coordinates
            if sideMaxX > maxX:
                maxX = sideMaxX
                maxXHeightSide = side

    return maxYWidthSide, maxXHeightSide



def findBestFitCircle(imageToDisplay, contour, xOffset, yOffset, borderPixels=10, maxIterations=10, initialDp=1, initialParam2=50,
                      minDist=10, minRadius=2, maxRadius=0):

    height, width = 720,1280
    blackImage = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.drawContours(blackImage, [contour], -1, (255, 255, 255), 2)
    blackImage = cv2.cvtColor(blackImage, cv2.COLOR_BGR2GRAY)



    # set initial parameters for Hough Circle detection
    dp = initialDp
    param2 = initialParam2

    # Iteratively attempt to detect a circle, relaxing parameters if needed
    for i in range(maxIterations):
        # Try detecting circles with current parameters
        circles = cv2.HoughCircles(blackImage, cv2.HOUGH_GRADIENT, dp, minDist, param1=50, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)

        # If at least one circle is found, take the first circle as the best fit
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]

            # Draw the detected circle for visualization

            #cv2.circle(blackImage, (x, y), r, (0, 255, 0), 2)  # Circle outline in green
            #cv2.circle(blackImage, (x, y), 2, (0, 0, 255), 3)  # Center in red
            cv2.circle(imageToDisplay, (x+xOffset, y+yOffset), 2, (0, 0, 255), 2)  # Center in red
            cv2.circle(imageToDisplay, (x+xOffset, y+yOffset), r, (0, 255, 0), 2)  # Circle outline in green


            # Adjust circle coordinates to account for the added border
            x -= borderPixels
            y -= borderPixels
            circleParameters = {'dp': dp, 'param2': param2}
            return r, circleParameters

        # Adjust parameters to make the detection less strict on subsequent iterations
        dp += 0.5  # Increase dp to allow larger accumulator resolution
        param2 = max(10, param2 - 5)  # Decrease param2 to allow easier circle detection (lower threshold)

    print("nothing found")
    return None, None



def findKnownCircles(imageToDisplay, contour, dimension, radiusTolerance=7, arcTolerance=8):
    height, width = 720, 1280
    blackImage = cv2.cvtColor(cv2.drawContours(
        np.zeros((height, width, 3), dtype=np.uint8), [contour], -1, (255, 255, 255), 2), cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        blackImage, cv2.HOUGH_GRADIENT, dimension.circleParams['dp'], 50, param1=50,
        param2=dimension.circleParams['param2'],
        minRadius=dimension.refRadius - radiusTolerance,
        maxRadius=dimension.refRadius + radiusTolerance
    )

    allCenters, allPoint2 = (0, 0), (0, 0)
    if circles is not None:
        #circle's center and radius
        x, y, r = np.round(circles[0, 0]).astype(int)


        # Apply the offset adjustment for the circle center
        x += 290  # Align circle center with the adjusted coordinate system
        dimension.centerHistory.append((x, y))
        dimension.radiusHistory.append(r)

        # Calculate the average center and radius
        avg_center = tuple(np.mean(dimension.centerHistory, axis=0).astype(int))
        avg_radius = int(np.mean(dimension.radiusHistory))
        #print(f"Adjusted Circle Center: {avg_center}, Radius: {avg_radius}")

        cv2.circle(imageToDisplay, avg_center, 2, (0, 0, 255), 2)  # Draw the center point

        # Adjust contour points to align with the circle's coordinate system
        adjusted_contour = contour.reshape(-1, 2).copy()
        adjusted_contour[:, 0] += 290  # Remove the x-offset
        #print(f"Adjusted Contour Points (first 5): {adjusted_contour[:5]}")



        # Find contour points near the circle (within tolerance)
        arc_points = []
        for point in adjusted_contour:
            dist_to_circle = np.sqrt((point[0] - avg_center[0])**2 + (point[1] - avg_center[1])**2)
            #print(f"Point: {point}, Distance to Circle: {dist_to_circle}, Tolerance Range: [{avg_radius - arcTolerance}, {avg_radius + arcTolerance}]")

            if avg_radius - arcTolerance <= dist_to_circle <= avg_radius + arcTolerance:
                arc_points.append(point)

        # Draw the arc if valid points are found
        if arc_points:

            angles = [
                np.arctan2(point[1] - avg_center[1], point[0] - avg_center[0])
                for point in arc_points
            ]

            angles = [np.degrees(angle if angle >= 0 else angle + 2 * np.pi) for angle in angles]

            # Sort angles to define the arc
            angles = sorted(angles)

            # angle wraparound cases weird full circles depending on placement
            start_angle = angles[0]
            end_angle = angles[-1]
            if end_angle - start_angle > 180:

                split_index = np.argmax(np.diff(angles) > 180) + 1
                angles = angles[split_index:] + [a + 360 for a in angles[:split_index]]
                start_angle = angles[0]
                end_angle = angles[-1]
            cv2.ellipse(imageToDisplay, avg_center, (avg_radius, avg_radius),
                        0, start_angle, end_angle, (0, 255, 0), 2)  # Green arc


        allCenters = avg_center
        allPoint2 = (int(avg_center[0] + avg_radius * np.cos(np.radians(270))),
                     int(avg_center[1] + avg_radius * np.sin(np.radians(270))))
    else:
        print('No circles detected')
    #cv2.line(imageToDisplay, allCenters, allPoint2, (0,255,0), 2)
    return allCenters, allPoint2

#the method below will look for 1 circle, the one below that looks for all circles that match, doesnt work good

# def findKnownCircles(imageToDisplay, contour, dimension, radiusTolerance=7):
#     height, width = 720, 1280
#     blackImage = cv2.cvtColor(cv2.drawContours(
#         np.zeros((height, width, 3), dtype=np.uint8), [contour], -1, (255, 255, 255), 2), cv2.COLOR_BGR2GRAY)
#
#     circles = cv2.HoughCircles(
#         blackImage, cv2.HOUGH_GRADIENT, dimension.circleParams['dp'], 50, param1=50,
#         param2=dimension.circleParams['param2'],
#         minRadius=dimension.refRadius - radiusTolerance,
#         maxRadius=dimension.refRadius + radiusTolerance
#     )
#     allCenters, allPoint2 = (0,0),(0,0)
#     if circles is not None:
#
#
#         x, y, r = np.round(circles[0, 0]).astype(int)
#         dimension.centerHistory.append((x + 290, y))
#         dimension.radiusHistory.append(r)
#
#     if dimension.centerHistory and dimension.radiusHistory:
#         avg_center = tuple(np.mean(dimension.centerHistory, axis=0).astype(int))
#         avg_radius = int(np.mean(dimension.radiusHistory))
#         cv2.circle(imageToDisplay, avg_center, 2, (0, 0, 255), 2)
#         cv2.circle(imageToDisplay, avg_center, avg_radius, (0, 255, 0), 2)
#
#         allCenters = avg_center
#         allPoint2 = (int(avg_center[0] + avg_radius * np.cos(45)), int(avg_center[1] + avg_radius * np.sin(45)))
#     elif circles is not None:
#         allCenters = allPoint2 = (0, 0)
#         cv2.circle(imageToDisplay, (x, y), 2, (0, 0, 255), 2)
#         cv2.circle(imageToDisplay, (x, y), r, (0, 255, 0), 2)
#
#     else:
#         print('no circles')
#     return allCenters, allPoint2

# def findKnownCircles(imageToDisplay, contour, dimension, radiusTolerance=4):
#     height, width = 720, 1280
#     blackImage = cv2.cvtColor(cv2.drawContours(
#         np.zeros((height, width, 3), dtype=np.uint8), [contour], -1, (255, 255, 255), 2), cv2.COLOR_BGR2GRAY)
#
#     circles = cv2.HoughCircles(
#         blackImage, cv2.HOUGH_GRADIENT, dimension.circleParams['dp'], 50, param1=50,
#         param2=dimension.circleParams['param2'] + 1,
#         minRadius=dimension.refRadius - radiusTolerance,
#         maxRadius=dimension.refRadius + radiusTolerance
#     )
#
#     if circles is not None:
#         # Compute the average center if history exists
#         avg_center = np.mean(dimension.centerHistory, axis=0).astype(int) if dimension.centerHistory else None
#
#         # Select the circle closest to the average center
#         closest_circle = None
#         min_distance = float('inf')
#
#         for circle in np.round(circles[0, :]).astype("int"):
#             x, y, r = circle
#             if avg_center is not None:
#                 distance = np.linalg.norm(np.array([x + 290, y]) - avg_center)
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_circle = (x, y, r)
#             else:
#                 # No previous average center, take the first circle by default
#                 closest_circle = (x, y, r)
#                 break
#
#         # Update history with the selected circle
#         if closest_circle is not None:
#             x, y, r = closest_circle
#             dimension.centerHistory.append((x + 290, y))
#             dimension.radiusHistory.append(r)
#
#     # Calculate moving average for center and radius
#     if dimension.centerHistory and dimension.radiusHistory:
#         avg_center = tuple(np.mean(dimension.centerHistory, axis=0).astype(int))
#         avg_radius = int(np.mean(dimension.radiusHistory))
#         cv2.circle(imageToDisplay, avg_center, 2, (0, 0, 255), 2)
#         cv2.circle(imageToDisplay, avg_center, avg_radius, (0, 255, 0), 2)
#
#         allCenters = avg_center
#         allPoint2 = (int(avg_center[0] + avg_radius * np.cos(45)), int(avg_center[1] + avg_radius * np.sin(45)))
#     else:
#         allCenters = allPoint2 = (0, 0)
#
#     return allCenters, allPoint2
