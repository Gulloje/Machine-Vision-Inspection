# Form implementation generated from reading ui file '.\uiFiles\measurementPopup.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.txtDimensionName = QtWidgets.QLineEdit(parent=Form)
        self.txtDimensionName.setGeometry(QtCore.QRect(120, 60, 132, 21))
        self.txtDimensionName.setObjectName("txtDimensionName")
        self.txtExpectedValue = QtWidgets.QLineEdit(parent=Form)
        self.txtExpectedValue.setGeometry(QtCore.QRect(120, 100, 132, 21))
        self.txtExpectedValue.setObjectName("txtExpectedValue")
        self.txtMinimumValue = QtWidgets.QLineEdit(parent=Form)
        self.txtMinimumValue.setGeometry(QtCore.QRect(120, 140, 132, 21))
        self.txtMinimumValue.setObjectName("txtMinimumValue")
        self.txtMaxValue = QtWidgets.QLineEdit(parent=Form)
        self.txtMaxValue.setGeometry(QtCore.QRect(120, 180, 132, 21))
        self.txtMaxValue.setObjectName("txtMaxValue")
        self.label = QtWidgets.QLabel(parent=Form)
        self.label.setGeometry(QtCore.QRect(120, 40, 151, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=Form)
        self.label_2.setGeometry(QtCore.QRect(120, 80, 151, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(parent=Form)
        self.label_3.setGeometry(QtCore.QRect(120, 120, 151, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(parent=Form)
        self.label_4.setGeometry(QtCore.QRect(120, 160, 151, 16))
        self.label_4.setObjectName("label_4")
        self.btnConfirm = QtWidgets.QPushButton(parent=Form)
        self.btnConfirm.setGeometry(QtCore.QRect(90, 220, 75, 24))
        self.btnConfirm.setObjectName("btnConfirm")
        self.btnCancel = QtWidgets.QPushButton(parent=Form)
        self.btnCancel.setGeometry(QtCore.QRect(200, 220, 75, 24))
        self.btnCancel.setObjectName("btnCancel")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt;\">Dimension Name</span></p></body></html>"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt;\">Expected Value</span></p></body></html>"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt;\">Minimum Value</span></p></body></html>"))
        self.label_4.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt;\">Maximum Value</span></p></body></html>"))
        self.btnConfirm.setText(_translate("Form", "Confirm"))
        self.btnCancel.setText(_translate("Form", "Cancel"))
