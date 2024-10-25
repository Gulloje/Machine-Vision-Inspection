from PyQt6 import QtCore, QtGui, QtWidgets
import sys, traceback
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

from DimensionDescriptor import DimensionDescriptor
from cameraThread import CameraThread
from cameraThread2 import CameraThread2
from cameraFeedUi import Ui_MainWindow
from popUpDialog import *
from masking import *


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        # Initalize camera
        self.camera = CameraThread2(self)
        self.camera.frameCaptured.connect(self.updateFrame)

        self.camera.start()





        # Initalize dimensions list and populate
        self.listModel = QtCore.QStringListModel() # for the static dimension options
        self.model = QtCore.QStringListModel() # for the user options
        self.listDimensionsChoice.setModel(self.listModel)
        self.populateDimensions()


        # store clicked position
        self.clickedPosition = None
        self.boundingBox = [None, None, None, None]
        self.drawing = False

        #set part Button
        self.btnSetPart.clicked.connect(self.setPart)
        self.radioLiveMeasure.toggled.connect(self.onLiveMeasure)
        self.radioSetDimensionMode.toggled.connect(self.onSetDimensionMode)

        #add dimension button
        self.btnAddDimension.clicked.connect(self.addDimension)

        #dropdown
        self.drpPartSelect.addItem("Part 1")
        self.drpPartSelect.addItem("Add Part")
        self.drpPartSelect.currentIndexChanged.connect(self.dropDownHandler)

        self.partDescriptor = []


    def updateFrame(self, image: np.ndarray):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        #convert to q image to display
        qImg = QtGui.QImage(image.data.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format.Format_BGR888)

        # Set the QImage in the QLabel
        self.camFeedOne.setPixmap(QtGui.QPixmap.fromImage(qImg))


    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.boundingBox = [None, None, None, None]
            widget_pos = self.camFeedOne.mapFromGlobal(event.globalPosition().toPoint())
            x = widget_pos.x()
            y = widget_pos.y()

            # Make sure the click is within the bounds of the QLabel
            if 0 <= x < self.camFeedOne.width() and 0 <= y < self.camFeedOne.height():
                self.clickedPosition = (x, y)
                # Pass the clicked position to the CameraThread for distance calculation
                self.boundingBox[0] = x
                self.boundingBox[1] = y
                self.drawing = True
                self.camera.updateClickPosition(self.clickedPosition)

    def mouseMoveEvent(self, event):
        if self.drawing:
            widget_pos = self.camFeedOne.mapFromGlobal(event.globalPosition().toPoint())
            x = widget_pos.x()
            y = widget_pos.y()
            if 0 <= x < self.camFeedOne.width() and 0 <= y < self.camFeedOne.height():
                self.boundingBox[2] = x
                self.boundingBox[3] = y
                self.drawing = True
                self.camera.updateBoxPosition(self.boundingBox)


    def setPart(self):
        try:
            colorImage, depthFrame = self.camera.getColorAndDepthImage()  # want to get the original (cropped)image, not anything edited or shrunk
            colorImage = cropImageBorder(colorImage)

            contours, contourLengths = getBinaryImage(colorImage)
            self.camera.currentContour = max(contours, key=cv2.contourArea)
            self.camera.stillImage=None
            self.camera.savedContourLengths = contourLengths
        except Exception as e:
            print(e)

    def addDimension(self):
        try:

            selectedIndex = self.listDimensionsChoice.selectionModel().currentIndex()
            selectedItem = ""
            if selectedIndex.isValid():
                selectedItem = self.listModel.data(selectedIndex)
                print(selectedItem)
            if selectedItem == "Height + Width":
                print('do a length')
                self.camera.setLengthAndWidth()
                self.createPopup("Width")
            elif selectedItem == "Diameter":
                print("bruh do a Diameter")
                self.camera.setDiameter()
            elif selectedItem == "Chamfer":
                print("bruh do a Chamfer")
            elif selectedItem == "Angle":
                print("bruh do a Angle")
            elif selectedItem == "Radius":
                print("bruh do a Radius")
            else:
                print("bruh select Something")

            self.createPopup(selectedItem)

        except Exception as e:
            print(e)





    def onLiveMeasure(self):
        if self.radioLiveMeasure.isChecked():
            print("Live measurement toggled on")
            self.camera.setLiveMeasureFlag(True)  # Enable live measurement
        else:
            print("Live measurement toggled off")
            self.camera.setLiveMeasureFlag(False)  # Disable live measurement

    def onSetDimensionMode(self):
        if self.radioSetDimensionMode.isChecked():
            print('set dimension is checked')
            self.camera.setDimensionIsOn = True
        else:
            print('set dimension toggled off')
            self.camera.setDimensionIsOn = False

    def dropDownHandler(self, index):
        if self.drpPartSelect.itemText(index) == "Add Part":
            self.drpPartSelect.blockSignals(True)
            newName, ok = QtWidgets.QInputDialog.getText(self, "Add Part", "Enter Part Name")
            if ok and newName:
                self.drpPartSelect.insertItem(self.drpPartSelect.count()-1, newName)
                self.drpPartSelect.setCurrentText(newName)
            self.drpPartSelect.blockSignals(False)

    def mouseReleaseEvent(self, event):
        if self.drawing:
            widget_pos = self.camFeedOne.mapFromGlobal(event.globalPosition().toPoint())
            x = widget_pos.x()
            y = widget_pos.y()
            if 0 <= x < self.camFeedOne.width() and 0 <= y < self.camFeedOne.height():
                self.boundingBox[2] = x
                self.boundingBox[3] = y
                self.drawing = False
                self.camera.updateBoxPosition(self.boundingBox)



    def populateDimensions(self):
        dimensions = ['Height + Width', 'Chamfer', 'Radius', 'Angle', 'Diameter']
        self.listModel.setStringList(dimensions)



    def closeEvent(self, event):
        #stop the camera thread too
        self.camera.stop()
        event.accept()

    def addSavedDimension(self, name, expected, min, max):


        self.listPartDimensions.setModel(self.model)
        list = self.model.stringList()
        list.append(name + ": " + expected + ", " + min+" - "+max + "mm")
        self.model.setStringList(list)

    def createPopup(self, selectedItem):
        if selectedItem =="Height + Width":
            selectedItem = "Height"
        popup = PopupDialog(self, windowTitle=str(selectedItem) + " Parameters")

        if popup.exec():
            self.addSavedDimension(popup.var1, popup.var2, popup.var3, popup.var4)
            contourLen = self.camera.contourDimensionLength
            contourArea = self.camera.contourArea
            dimension = DimensionDescriptor(selectedItem, popup.var1, popup.var2, popup.var3, popup.var4, contourLen,
                                            contourArea)
            self.partDescriptor.append(dimension)
            self.camera.partDescriptor = self.partDescriptor
            print(self.partDescriptor)
            self.camera.cleanImage = self.camera.stillImage.copy()
        else:
            self.camera.stillImage = self.camera.cleanImage.copy()
            print("Dialog was cancelled")


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
