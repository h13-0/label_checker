# import PyQt5
from PyQt6.QtWidgets import QApplication, QMainWindow

# import UI
from QtUI.LabelCheckerUI import LabelCheckerUI

import os
import sys
import threading
import logging
import datetime

from Workflow.MainWorkflow import MainWorkingFlow


if __name__ == '__main__':
    # 初始化退出信号
    stop_event = threading.Event()

    # 创建日志输出目录
    if not os.path.exists("log"):
        os.mkdir("log")

    # 设置日志等级
    logging.basicConfig(
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.DEBUG,
        #filename="log/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log",
        #filemode="a"
    )

    # 初始化UI
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = LabelCheckerUI()
    ui.setupUi(mainWindow)
    mainWindow.show()

    # 初始化工作流
    workflow = MainWorkingFlow(ui, stop_event)
    workflow.Run()

    ret = app.exec()
    stop_event.set()
    sys.exit(ret)
