import traceback

import cv2
import pyrealsense2 as rs
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
import math
from masking import *
from measurementFunctions import *


class CameraThread(QtCore.QThread):
    frameCaptured = QtCore.pyqtSignal(np.ndarray)  # signals to the ui to show a frame
    distanceCaptured = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = rs.pipeline()  # what owns the handles to the camera
        # self.pipeline.start(self.getConfigForPipeline())
        self.running = True
        self.isLiveMeasureOn = False
        self.savedContourLengths = []

        alignment = rs.stream.color
        self.align = rs.align(alignment)

        self.intrinsics = None
        # helps process the image to hopefully reduce noise
        self.spatialFilter = rs.spatial_filter()
        self.temporalFilter = rs.temporal_filter()

        self.depthFrame = None
        self.colorImage = None
        self.colorFrame = None
        self.wholeImage = None

        # for testing
        # helper for testing with mouse click
        self.point = (400, 300)
        self.boundingBox = [None, None, None, None]

    def getConfigForPipeline(self):
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        return config

    def setLiveMeasureFlag(self, flag):
        """Method to enable or disable live measurement."""
        self.isLiveMeasureOn = flag

    def run(self):

        try:
            self.running = True
            self.pipeline.start(self.getConfigForPipeline())

            while True:
                frames = self.pipeline.wait_for_frames()

                alignedFrames = self.align.process(frames)
                depthFrame = alignedFrames.get_depth_frame()
                colorFrame = alignedFrames.get_color_frame()

                filteredDepthFrame = self.spatialFilter.process(depthFrame)
                filteredDepthFrame = self.temporalFilter.process(filteredDepthFrame)
                if not depthFrame or not colorFrame or not filteredDepthFrame:
                    print('not getting frame')
                    continue

                self.intrinsics = colorFrame.profile.as_video_stream_profile().intrinsics

                depthImage = np.asanyarray(filteredDepthFrame.get_data())
                self.depthFrame = depthFrame
                self.colorFrame = colorFrame
                colorImage = np.asanyarray(colorFrame.get_data())
                self.wholeImage = colorImage

                cropped = colorImage[self.boundingBox[1]:self.boundingBox[3], self.boundingBox[0]:self.boundingBox[2]]
                self.colorImage = cropped

                cv2.circle(colorImage, self.point, 4, (0, 255, 0), -1)
                distance = depthImage[self.point[1], self.point[0]]
                if distance == 0:
                    pass
                if self.isLiveMeasureOn:
                    height, width = self.wholeImage.shape[:2]  # Get the dimensions of the image

                    # Crop the image: 50 pixels from the left, and 20 pixels from the right
                    cropped_image = self.wholeImage[:, 290:width - 90]
                    # Convert the current frame to grayscale
                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

                    blur = cv2.medianBlur(gray, 11)
                    thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                    cv2.imshow("Grayscale Live Feed", opening)

                    # Find contours in the processed image
                    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                    # Iterate over each contour detected in the current frame
                    for i, contour in enumerate(contours):
                        # Calculate the length of the current contour
                        contour_length = cv2.contourArea(contour, False)

                        # Check if this contour is close in size to any of the saved contours
                        for saved_length in self.savedContourLengths:
                            length_ratio = contour_length / saved_length

                            # If the length is similar (within 10%), draw a bounding box
                            if 0.9 < length_ratio < 1.1:  # 10 % threshhold
                                box = self.drawBoundingBox(self.wholeImage, contour)
                                cv2.drawContours(self.wholeImage, [box], 0, (0, 255, 0), 2)
                                length = self.calcDistance(box, 0, 1, i)
                                width = self.calcDistance(box, 1, 2, i)
                                drawMeasurements(self.wholeImage, box, length, width)

                if self.boundingBox[3] is not None:
                    cv2.rectangle(self.wholeImage,
                                  (int(self.boundingBox[0]), int(self.boundingBox[1])),
                                  (int(self.boundingBox[2]), int(self.boundingBox[3])),
                                  (0, 255, 0), 2)
                cv2.waitKey(1)

                self.frameCaptured.emit(self.wholeImage)

        except Exception as e:
            print(e)
            traceback.print_exc()

    def drawBoundingBox(self, image, contour):
        # Get bounding box coordinates for the current contour

        (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
        box = cv2.boxPoints(((x1 + 2, y1 + 2), (w1, h1), rotation))
        box = np.intp(box)
        box[:, 0] += 290  # Since the image is cropped from the left, add the offset

        # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        # Optionally, show the original full image with the corrected bounding box
        cv2.imshow("Original Image with Bounding Box", image)
        return box

    def updateClickPosition(self, point):
        self.point = point

    def getColorAndDepthImage(self):
        return self.colorImage, self.depthFrame

    def updateBoxPosition(self, boundingBox):
        self.boundingBox = boundingBox
        # print('draw the box')

    def stop(self):
        self.running = False  # Set flag to stop the thread loop
        self.wait()  # Wait for the thread to finish
        self.pipeline.stop()

    def calcLength(self, contours):
        imcopy = self.wholeImage.copy()
        for i, contour in enumerate(contours, start=1):
            offsetBox = self.getOffsetBoundingBox(contour)
            # print(offsetBox)
            length = self.calcDistance(offsetBox, 0, 1, i)
            width = self.calcDistance(offsetBox, 1, 2, i)
            print(f"{i}. Length: {length:.2f} mm")
            print(f"Width: {width:.2f} mm")

            # get midpoints in pixel length for printing on the picture
            # drawMeasurements(imcopy, offsetBox, length, width)

        cv2.drawContours(imcopy, [offsetBox], 0, (0, 0, 255), 2)
        cv2.circle(imcopy, tuple(offsetBox[0]), 3, (255, 0, 0), 2)  # min x blu
        cv2.circle(imcopy, tuple(offsetBox[1]), 3, (0, 255, 0), 2)  # min y green
        cv2.circle(imcopy, tuple(offsetBox[2]), 3, (0, 0, 255), 2)  # max x red
        cv2.circle(imcopy, tuple(offsetBox[3]), 3, (255, 255, 255), 2)  # max y white
        # cv2.imshow("Image", imcopy)   #IMPORTANT FOR DEBUGGING

    def getOffsetBoundingBox(self, contour):
        # this will help when trying to rotate the rectangle
        (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
        box = cv2.boxPoints(((x1 + 2, y1 + 2), (w1, h1),
                             rotation))  # i have the + 2 because of cropping the image when finding the contours, makes align better i think
        box = np.intp(box)
        offsetBox = np.array([[pt[0] + self.boundingBox[0], pt[1] + self.boundingBox[1]] for pt in box],
                             dtype=np.int32)
        return offsetBox

    def calcDistance(self, offsetBox, point1Index, point2Index, contourIndex):
        point1, point2 = offsetBox[point1Index], offsetBox[point2Index]
        # print(point1, point2)

        h = self.depthFrame.get_height()
        w = self.depthFrame.get_width()
        udist, vdist = 0, 0
        # Check if point1 is within valid bounds
        if 0 <= point1[0] < w and 0 <= point1[1] < h:
            udist = self.depthFrame.get_distance(point1[0], point1[1])  # gets the depth data at the pixel point
        if 0 <= point2[0] < w and 0 <= point2[1] < h:
            vdist = self.depthFrame.get_distance(point2[0], point2[1])

        if udist == 0 or vdist == 0:  # should check that the addition is within the frame
            print(f"Contour {contourIndex}: Depth data missing for some corners")
            # point1[1] += 50
            # point2[1] += 50
            # udist = self.depthFrame.get_distance(point1[0], point1[1])
            # vdist = self.depthFrame.get_distance(point2[0], point2[1])

        # Deproject the points to 3D space
        point1_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point1[0], point1[1]], udist)
        point2_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point2[0], point2[1]], vdist)
        # print(point1_3d, point2_3d)

        distMeters = math.sqrt(
            math.pow(point1_3d[0] - point2_3d[0], 2)
            + math.pow(point1_3d[1] - point2_3d[1], 2)
            + math.pow(point1_3d[2] - point2_3d[2], 2))

        distMM = distMeters * 1000

        return distMM
