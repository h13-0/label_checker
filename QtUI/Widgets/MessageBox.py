import threading
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtWidgets import QMessageBox

class MessageBox(QObject):
    _signal = pyqtSignal()
    def __init__(self, 
        parent, 
        title:str, 
        content:str, 
        icon:QMessageBox.Icon,
        button:QMessageBox.StandardButton=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        default_button:QMessageBox.StandardButton=QMessageBox.StandardButton.Yes
    ) -> None:
        super().__init__()
        self._parent = parent
        self._title = title
        self._content = content
        self._icon = icon
        self._button = button
        self._default_button = default_button


    @pyqtSlot()
    def _make_msg_box(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(self._title)
        msg_box.setText(self._content)
        msg_box.setIcon(self._icon)
        msg_box.setStandardButtons(self._button)
        msg_box.setDefaultButton(self._default_button)
        msg_box.exec()


    def exec(self):
        print(f"Current thread name: {threading.current_thread().name}")
        if(threading.current_thread().name == "MainThread"):
            self._make_msg_box()
        else:
            self._signal.connect(self._make_msg_box)
            self._signal.emit()
