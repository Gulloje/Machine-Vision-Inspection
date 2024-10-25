from PyQt6 import QtWidgets
from measurementPopup import Ui_Form  # Assuming Ui_Form is the generated class from the popup UI file

class PopupDialog(QtWidgets.QDialog, Ui_Form):
    def __init__(self, parent=None, windowTitle="Dimension Parameters"):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(windowTitle)
        self.btnConfirm.clicked.connect(self.confirm)
        self.btnCancel.clicked.connect(self.cancel)
        self.input_field = None

        self.var1 = 0
        self.var2 = 0
        self.var3 = 0
        self.var4 = 0

    #def get_input(self):
        #return self.input_field.text()

    def confirm(self):
        self.var1 = self.txtDimensionName.text()
        self.var2 = self.txtExpectedValue.text()
        self.var3 = self.txtMinimumValue.text()
        self.var4 = self.txtMaxValue.text()

        print("test")
        self.accept()

    def cancel(self):
        self.reject()



