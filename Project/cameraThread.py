import traceback

import cv2
import pyrealsense2 as rs
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from SelectContourUI import *
import math
from masking import *
from measurementFunctions import *


class CameraThread2(QtCore.QThread):
    frameCaptured = QtCore.pyqtSignal(np.ndarray)  # signals to the ui to show a frame
    distanceCaptured = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = rs.pipeline()  # what owns the handles to the camera
        # self.pipeline.start(self.getConfigForPipeline())
        self.running = True
        self.isLiveMeasureOn = False

        alignment = rs.stream.color
        self.align = rs.align(alignment)

        self.intrinsics = None
        # helps process the image to hopefully reduce noise
        self.spatialFilter = rs.spatial_filter()
        self.temporalFilter = rs.temporal_filter()

        self.depthFrame = None
        self.croppedColorImage = None
        self.colorFrame = None
        self.wholeImage = None
        self.stillImage = None
        self.cleanImage = None
        self.imageForDeletion = None

        # for testing
        # helper for testing with mouse click
        self.point = (400, 300)
        self.boundingBox = [None, None, None, None]

        self.frameBufferCount = 0  # Counter for frames
        self.previousFrame = None  # To store the previous frame for movement detection
        self.movementThreshold = 50  # Threshold for detecting significant movement
        self.savedContourLengths = []
        self.currentContour = None

        self.frameEmitted = False
        self.setDimensionIsOn = False
        self.partDescriptor = []

        #these are updated when setting dimensions in main.py, unsure if need to update anywhere else
        self.measurementBuffers = {dimension.name: [] for dimension in self.partDescriptor}  # dict of separate lists for each dimension
        self.averageMeasurements = {dimension.name: None for dimension in self.partDescriptor}



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

            while self.running:
                frames = self.pipeline.wait_for_frames()

                alignedFrames = self.align.process(frames)
                depthFrame = alignedFrames.get_depth_frame()
                colorFrame = alignedFrames.get_color_frame()

                filteredDepthFrame = self.spatialFilter.process(depthFrame)
                filteredDepthFrame = self.temporalFilter.process(filteredDepthFrame)
                if not depthFrame or not colorFrame or not filteredDepthFrame:
                    print('Not getting frame')
                    continue

                self.intrinsics = colorFrame.profile.as_video_stream_profile().intrinsics


                self.depthFrame = depthFrame
                self.colorFrame = colorFrame
                colorImage = np.asanyarray(colorFrame.get_data())
                self.wholeImage = colorImage  # Full original image

                cropped = colorImage[self.boundingBox[1]:self.boundingBox[3], self.boundingBox[0]:self.boundingBox[2]]
                self.croppedColorImage = cropped  # Cropped image for measurement

                # Check if live measurement is enabled
                if self.isLiveMeasureOn:
                    self.liveMeasure(colorImage)  # Perform live measurement and display results
                    # emits the image in the calcLength function
                elif self.setDimensionIsOn:
                    if self.stillImage is None:
                        self.stillImage = colorImage

                        #take outside of if statement if want a live feed
                        self.cleanImage = self.stillImage.copy()
                        self.imageForDeletion = self.stillImage.copy()
                    self.emitTheRectangle(self.stillImage)

                else:
                    self.frameEmitted = False
                    # If live measurement is not enabled, just display the active camera feed
                    if self.boundingBox[3] is not None:
                        cv2.rectangle(colorImage,
                                      (int(self.boundingBox[0]), int(self.boundingBox[1])),
                                      (int(self.boundingBox[2]), int(self.boundingBox[3])),
                                      (0, 255, 0), 2)

                    self.frameCaptured.emit(colorImage)


        except Exception as e:
            print(e)
            traceback.print_exc()

    def updateClickPosition(self, point):
        self.point = point

    def emitTheRectangle(self, img):
        img = img.copy()
        if self.boundingBox[3] is not None:
            cv2.rectangle(img, (int(self.boundingBox[0]), int(self.boundingBox[1])),
                          (int(self.boundingBox[2]), int(self.boundingBox[3])), (0, 255, 0), 2)
        self.frameCaptured.emit(img)


    def getColorAndDepthImage(self):
        return self.croppedColorImage, self.depthFrame

    def updateBoxPosition(self, boundingBox):
        self.boundingBox = boundingBox
        # print('draw the box')

    def stop(self):
        self.running = False  # Set flag to stop the thread loop
        self.wait()  # Wait for the thread to finish
        self.pipeline.stop()

    def liveMeasure(self, colorImage):
        image, opening, contours = processLiveFeed(colorImage)
        #cv2.imshow('Filtered', opening)
        #cv2.waitKey(1)
        if contours and compareContours(self.currentContour, max(contours, key=cv2.contourArea)) < .05:
            self.frameBufferCount += 1
            textYOffset = 0
            for contour in contours:
                contourLen = cv2.arcLength(contour, True)
                contourArea = cv2.contourArea(contour)
                for dimension in self.partDescriptor:
                    tolerance = 0.08
                    if contourLen < 75 or dimension.type == "Radius":
                        tolerance = 0.15
                    lenLower = dimension.contourArcLen * (1 - tolerance)
                    lenUpper = dimension.contourArcLen * (1 + tolerance)
                    areaLower = dimension.contourArea * (1 - tolerance)
                    areaUpper = dimension.contourArea * (1 + tolerance)

                    if (lenLower <= contourLen <= lenUpper) and (areaLower <= contourArea <= areaUpper):

                        point1, point2 = drawCorrespondingDimension(dimension, self.wholeImage, contour, 290, 0)
                        measurement = self.measureDistBetweenTwoPoint(point1, point2)
                        if  dimension.type == 'Diameter' and measurement != 0:
                            #print('getting added')
                            measurement += .020
                        if dimension.type == "Diameter": #draw a line through a diameter
                            cv2.line(self.wholeImage, point1,  point2, (0, 255, 0), 2)

                        if .0021 < measurement <= 300:  # Valid measurement
                            self.measurementBuffers[dimension.name].append(measurement)

                        # Check if buffer for the current dimension has reached 30 frames
                        if len(self.measurementBuffers[dimension.name]) == 30:
                            avg_measurement = sum(self.measurementBuffers[dimension.name]) / len(self.measurementBuffers[dimension.name])
                            print(f"Average Measurement for {dimension.name} over 30 frames: ", avg_measurement)
                            self.averageMeasurements[dimension.name] = avg_measurement  # Store averaged measurement for display
                            self.measurementBuffers[dimension.name] = []  # Reset buffer after averaging

                        # Display averaged measurement if calculated else display live measurement
                        if self.averageMeasurements[dimension.name] is not None:
                            displayMeasurements(self.wholeImage, self.averageMeasurements[dimension.name],
                                                dimension,
                                                textYOffset)
                        else:
                            displayMeasurements(self.wholeImage, measurement, dimension, textYOffset)

                        textYOffset += 30


        self.frameCaptured.emit(self.wholeImage)



    def measureDistBetweenTwoPoint(self, point1, point2):
        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]), int(point2[1]))

        h = self.depthFrame.get_height()
        w = self.depthFrame.get_width()
        udist, vdist = 0, 0
        if 0 <= point1[0] < w and 0 <= point1[1] < h:
            udist = self.depthFrame.get_distance(point1[0], point1[1])  # gets the depth data at the pixel point
            #print(point1[0], point1[1], ": ", udist)
        if 0 <= point2[0] < w and 0 <= point2[1] < h:
            vdist = self.depthFrame.get_distance(point2[0], point2[1])
            #print(point2[0], point2[1], ": ", vdist)
        if udist == 0 or vdist == 0:  # should check that the addition is within the frame
            print(f" Depth data missing for udist")
            udist = .317
        if vdist ==0:
            print(f" Depth data missing for vdist")
            vdist = .317


        point1_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point1[0], point1[1]], udist)
        point2_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point2[0], point2[1]], vdist)
        distMeters = math.sqrt(
            math.pow(point1_3d[0] - point2_3d[0], 2)
            + math.pow(point1_3d[1] - point2_3d[1], 2)
            + math.pow(point1_3d[2] - point2_3d[2], 2))

        distMM = distMeters * 1000
        distIn = distMM / 25.4

        return distIn


    def detectMovement(self, previous_frame, current_frame):
        frameDifference = cv2.absdiff(previous_frame, current_frame)

        threshold = 25
        _, thresh = cv2.threshold(frameDifference, threshold, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if motion is detected
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Adjust this threshold for sensitivity
                motion_detected = True
                break

        return motion_detected



    #crops by where the bounding box is
    def cropImageByBoundingBox(self, image):
        cropped = image[self.boundingBox[1]:self.boundingBox[3], self.boundingBox[0]:self.boundingBox[2]]
        return cropped

    def setDimension(self, dimension):
        img = self.cropImageByBoundingBox(self.imageForDeletion.copy())
        img, opening, contours = processLiveFeed(img)
        handler = ContourHandler(img, contours, dimension.type.lower())
        cv2.namedWindow("Choose " + dimension.type)
        handler.set_mouse_callback("Choose " + dimension.type)
        handler.wait_for_key("Choose " + dimension.type)

        if (handler.selected_contour is not None):
            dimension.contourArcLen = cv2.arcLength(handler.selected_contour, True)
            dimension.contourArea = cv2.contourArea(handler.selected_contour)
        if dimension.type == "Diameter":
            drawCircleFromMinRect(self.stillImage, handler.selected_contour, self.boundingBox[0], self.boundingBox[1])
        elif dimension.type == "Radius":
            partialContour = handler.get_contour_segment_between_points()
            dimension.refRadius, dimension.circleParams = findBestFitCircle(self.stillImage, partialContour, self.boundingBox[0], self.boundingBox[1])

        else:
            pt1, pt2 = drawMinAreaRect(self.stillImage, handler.selected_contour, self.boundingBox[0], self.boundingBox[1], dimension.type)
            if dimension.type == "Height":
                dimension.heightRefLen = math.dist(pt1, pt2)
            else:
                dimension.widthRefLen = math.dist(pt1, pt2)


