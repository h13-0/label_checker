import threading

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QInputDialog

class InputDialog(QInputDialog):
    _signal = pyqtSignal()
    def __init__(self,
        parent,
        title:str,
        label_text:str,
        input_mode:QInputDialog.InputMode=QInputDialog.InputMode.TextInput,
        text_value:str=""
    ) -> None:
        self._parent = parent
        self._title = title
        self._label_text = label_text
        self._input_mode = input_mode
        self._text_value = text_value
        self._ok_pressed = False

        self._finish = threading.Condition()

    @pyqtSlot()
    def _make_input_dialog(self):
        dialog = QInputDialog()
        dialog.setWindowTitle(self._title)
        dialog.setLabelText(self._label_text)
        dialog.setInputMode(self._input_mode)
        dialog.setTextValue(self._text_value)

        self._ok_pressed = dialog.exec()
        self._text_value = dialog.textValue()
        with self._finish:
            self._finish.notify()


    def exec(self):
        if(threading.current_thread().name == "MainThread"):
            self._make_input_dialog()
        else:
            self._signal.connect(self._make_input_dialog)
            self._signal.emit()
            with self._finish:
                self._finish.wait()


    def ok_pressed(self) -> bool:
        return self._ok_pressed
    

    def get_input_text(self) -> str:
        return self._text_value
    
    