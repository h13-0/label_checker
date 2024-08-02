from QtUI.Ui_TemplateEditor import Ui_TemplateEditor
from PyQt6.QtWidgets import QWidget


class TemplateEditorUI(Ui_TemplateEditor, QWidget):
    def __init__(self):
        Ui_TemplateEditor.__init__(self)
        QWidget.__init__(self)


    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

