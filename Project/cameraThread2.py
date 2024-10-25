import traceback

import cv2
import pyrealsense2 as rs
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from FindLengthWindow import *
import math
from masking import *
from measurementFunctions import *


class CameraThread2(QtCore.QThread):
    frameCaptured = QtCore.pyqtSignal(np.ndarray)  # signals to the ui to show a frame
    distanceCaptured = QtCore.pyqtSignal(str)
    contourDimensionLength = 0
    contourArea = 0

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

                depthImage = np.asanyarray(filteredDepthFrame.get_data())
                self.depthFrame = depthFrame
                self.colorFrame = colorFrame
                colorImage = np.asanyarray(colorFrame.get_data())
                self.wholeImage = colorImage  # Full original image

                cropped = colorImage[self.boundingBox[1]:self.boundingBox[3], self.boundingBox[0]:self.boundingBox[2]]
                self.croppedColorImage = cropped  # Cropped image for measurement

                # Check if live measurement is enabled
                if self.isLiveMeasureOn:
                    #self.liveMeasure(colorImage, depthImage)  # Perform live measurement and display results
                    self.liveMeasure2(colorImage)  # Perform live measurement and display results

                    # emits the image in the calcLength function
                elif self.setDimensionIsOn:
                    if self.stillImage is None:
                        self.stillImage = colorImage

                        #take outside of if statement if want a live feed
                        savedImage, filteredImage, contours = processLiveFeed(self.stillImage.copy())
                        #drawOffsetContours(self.stillImage, contours)
                        self.cleanImage = self.stillImage.copy()
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

    def getOffsetBoundingBox(self, contour, xOffset, yOffset):
        # this will help when trying to rotate the rectangle
        (x1, y1), (w1, h1), rotation = cv2.minAreaRect(contour)
        box = cv2.boxPoints(((x1 + 2, y1 + 2), (w1, h1),
                             rotation))  # i have the + 2 because of cropping the image when finding the contours, makes align better i think
        box = np.intp(box)
        offsetBox = np.array([[pt[0] + xOffset, pt[1] + yOffset] for pt in box],
                             dtype=np.int32)
        return offsetBox



    def liveMeasure(self, colorImage, depthImage):
        """Main method to capture measurements and detect movement."""
        image, opening, contours = processLiveFeed(colorImage)

        cv2.imshow('live filtered feed', opening)
        cv2.waitKey(1)


        # Check for movement by comparing with the previous frame
        if self.previousFrame is not None:
            movement_detected = self.detectMovement(self.previousFrame, opening)

            if movement_detected:
                print("Movement detected. Resetting measurements...")
                self.frameBufferCount = 0
                self.previousFrame = opening  # Store current frame for next iteration
                self.frameCaptured.emit(colorImage)
                self.frameEmitted = False
                return  # Exit early if movement is detected

        # Only accumulate frames and process contours if no movement is detected

        if contours:
            self.frameBufferCount += 1  # Increment the frame buffer count only if no movement
            for contour in contours:
                contourLength = cv2.contourArea(contour, False)
                savedContourLengths_np = np.array(self.savedContourLengths)
                lengthRatios = contourLength / savedContourLengths_np

                # Use any() to check if any length ratio is within the desired threshold
                if np.any((lengthRatios > 0.9) & (
                        lengthRatios < 1.1)) and self.frameBufferCount >= 60 and not self.frameEmitted:
                    #self.calcLength(contours)
                    self.frameEmitted = True
                    self.frameCaptured.emit(self.wholeImage)
                    # Reset for the next cycle
                    self.frameBufferCount = 0

        # Update the previous frame for the next iteration
        self.previousFrame = opening  # Store current frame for next iteration


    def liveMeasure2(self, colorImage):
        image, opening, contours = processLiveFeed(colorImage)

        #cv2.imshow('live filtered feed', opening) #idk why this is crashing it all of a suddem
        #cv2.waitKey(1)


        if contours and compareContours(self.currentContour, max(contours, key=cv2.contourArea)) < .05:
            self.frameBufferCount += 1  # Increment the frame buffer count only if no movement
            textYOffset = 0
            for contour in contours:
                contourLen = cv2.arcLength(contour, True)
                contourArea = cv2.contourArea(contour)
                for dimension in self.partDescriptor:
                    tolerance = .05 # percent COMEBACK: might have to play with tolerance
                    if contourLen < 75:  # up the tolerance for smaller contours
                        tolerance = .15
                    lenLower = dimension.contourArcLen * (1-tolerance)
                    lenUpper = dimension.contourArcLen * (1+tolerance)

                    areaLower = dimension.contourArea * (1 - tolerance)
                    areaUpper = dimension.contourArea * (1 + tolerance)

                    if (lenLower <= contourLen <= lenUpper) and (areaLower <= contourArea <= areaUpper):

                        point1, point2 = drawCorrespondingDimension(dimension, self.wholeImage, contour, 290, 0)
                        measurement = self.measureDistBetweenTwoPoint(point1, point2)
                        #if measurement == 0 or measurement > 300: #means it didnt pick up the depth data correctly
                            #continue
                        print("measurement = ", measurement)
                        displayMeasurements(self.wholeImage, measurement, dimension, textYOffset)
                        textYOffset += 30
        self.frameCaptured.emit(self.wholeImage)


    def measureDistBetweenTwoPoint(self, point1, point2):
        point1 = (int(point1[0]), int(point1[1]))
        point2 = (int(point2[0]), int(point2[1]))
        cv2.line(self.wholeImage,point1, point2, (0,255,0), 2)
        h = self.depthFrame.get_height()
        w = self.depthFrame.get_width()
        udist, vdist = 0, 0
        if 0 <= point1[0] < w and 0 <= point1[1] < h:
            udist = self.depthFrame.get_distance(point1[0], point1[1])  # gets the depth data at the pixel point
        if 0 <= point2[0] < w and 0 <= point2[1] < h:
            vdist = self.depthFrame.get_distance(point2[0], point2[1])

        if udist == 0 or vdist == 0:  # should check that the addition is within the frame
            print(f" Depth data missing for some corners")

        point1_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point1[0], point1[1]], udist)
        point2_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [point2[0], point2[1]], vdist)
        distMeters = math.sqrt(
            math.pow(point1_3d[0] - point2_3d[0], 2)
            + math.pow(point1_3d[1] - point2_3d[1], 2)
            + math.pow(point1_3d[2] - point2_3d[2], 2))

        distMM = distMeters * 1000

        return distMM


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


    def setLengthAndWidth(self):
        img = self.cropImageByBoundingBox(self.cleanImage.copy())
        img, opening, contours = processLiveFeed(img)
        handler = ContourHandler(img, contours)
        cv2.namedWindow("Choose Length")
        handler.set_mouse_callback("Choose Length")
        handler.wait_for_key("Choose Length")

        if (handler.selected_contour is not None):
            self.contourDimensionLength = cv2.arcLength(handler.selected_contour, True)
            self.contourArea = cv2.contourArea(handler.selected_contour)
            drawMinAreaRect(self.stillImage, handler.selected_contour, self.boundingBox[0], self.boundingBox[1])

    def setDiameter(self):
        img = self.cropImageByBoundingBox(self.cleanImage.copy())
        img, opening, contours = processLiveFeed(img)
        handler = ContourHandler(img, contours)
        cv2.namedWindow("Choose Diameter")
        handler.set_mouse_callback("Choose Diameter")
        handler.wait_for_key("Choose Diameter")

        self.contourDimensionLength = cv2.arcLength(handler.selected_contour, True)
        self.contourArea = cv2.contourArea(handler.selected_contour)
        drawCircleFromMinRect(self.stillImage, handler.selected_contour, self.boundingBox[0], self.boundingBox[1])

