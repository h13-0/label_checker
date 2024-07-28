# import PyQt5
from PyQt6.QtWidgets import QApplication, QMainWindow

# import UI
from QtUI.UI_Child import Ui_Main

import sys
import threading

from Workflow.MainWorkflow import MainWorkingFlow


if __name__ == '__main__':
    # 退出信号
    stop_event = threading.Event()

    # 初始化UI
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_Main()
    ui.setupUi(mainWindow)
    mainWindow.show()

    # 初始化工作流
    workflow = MainWorkingFlow(ui, stop_event)
    workflow.Run()

    ret = app.exec()
    stop_event.set()
    sys.exit(ret)
