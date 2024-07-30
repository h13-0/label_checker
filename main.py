# import PyQt5
from PyQt6.QtWidgets import QApplication, QMainWindow

# import UI
from QtUI.UI_Child import Ui_Main

import sys
import threading
import logging

from Workflow.MainWorkflow import MainWorkingFlow


if __name__ == '__main__':
    # 初始化退出信号
    stop_event = threading.Event()

    # 设置日志等级
    logging.basicConfig(level=logging.DEBUG)

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
