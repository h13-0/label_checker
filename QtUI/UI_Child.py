from QtUI import UI
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout
from PyQt6.QtGui import QPixmap, QImage
import logging
from enum import Enum
import cv2


class CallbackType(Enum):
    OpenTemplateClicked = 1,
    OpenTargetStreamClicked = 2,
    OpenTargetPhotoClicked = 3,
    LoadParamsClicked = 4


"""
@brief: 窗口的常驻图像控件(GraphicView)的枚举类
"""
class GraphicWidgets(Enum):
    MainGraphicView = 1,
    TemplateGraphicView = 2


class Ui_Main(UI.Ui_LabelChecker, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, GraphicWidgets)
    _set_graphic_detail_signal = pyqtSignal(dict)

    def __init__(self):
        UI.Ui_LabelChecker.__init__(self)
        QWidget.__init__(self)
        self._callback_map = {}

    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        # 初始化GraphicDetialList
        self.GraphicDetialListContents = QWidget()
        self.GraphicDetialListvLayout = QVBoxLayout(self.GraphicDetialListContents)
        self.GraphicDetialListContents.setLayout(self.GraphicDetialListvLayout)
        self.GraphicDetialList.setWidget(self.GraphicDetialListContents)

        # 常驻GraphicView的控件及其scene
        self._graphic_widgets = {
            "MainGraphicView": self.MainGraphicView,
            "TemplateGraphicView": self.TemplateGraphicView
        }
        self._graphic_widgets_scenes = {
            "MainGraphicView": QGraphicsScene(),
            "TemplateGraphicView": QGraphicsScene()
        }
        for widgets in self._graphic_widgets_scenes:
            self._graphic_widgets[widgets].setScene(
                self._graphic_widgets_scenes[widgets]
            )

        # 连接回调函数
        self.OpenTemplateButton.clicked.connect(lambda:self._callbacks(CallbackType.OpenTemplateClicked))
        self.OpenTargetStreamButton.clicked.connect(lambda:self._callbacks(CallbackType.OpenTargetStreamClicked))
        self.OpenTargetPhotoButton.clicked.connect(lambda:self._callbacks(CallbackType.OpenTargetPhotoClicked))
        self.LoadParamsButton.clicked.connect(lambda:self._callbacks(CallbackType.LoadParamsClicked))

        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._set_graphic_detail_signal.connect(self._set_graphic_detail_content, type=Qt.ConnectionType.BlockingQueuedConnection)

    def _callbacks(self, target:CallbackType):
        if(self._callback_map[target.name] is None):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._callback_map[target.name]()


    '''
    @brief: 更新图像的槽函数
    '''
    def _update_graphic_view(self, img:QImage, target:GraphicWidgets):
        pixmap = QPixmap.fromImage(img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self._graphic_widgets_scenes[target.name].clear()
        self._graphic_widgets_scenes[target.name].addItem(pixmap_item)
        self._graphic_widgets_scenes[target.name].update()


    """
    @brief: 设置 "图像详情列表" 元素的槽函数
    @param:
        - details:
            {
                "title": QImage
            }
    """
    def _set_graphic_detail_content(self, details:dict):
        # 清除UI中所有现存图像
        while self.GraphicDetialListvLayout.count():
            item = self.GraphicDetialListvLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 绘制新图像
        for title in details:
            img = details[title]
            img_w = img.width()
            img_h = img.height()
            # 创建GroupBox
            group_box = QtWidgets.QGroupBox(title, parent=self.GraphicDetialList)
            group_box.setMinimumSize(QtCore.QSize(img_w + 20, img_h + 30))
            group_box.setMaximumSize(QtCore.QSize(img_w + 20, img_h + 30))
            self.GraphicDetialListvLayout.addWidget(group_box)
        
            # 创建scene
            scene = QGraphicsScene()
            # 创建GraphicView
            graphic_view = QtWidgets.QGraphicsView(group_box)
            graphic_view.setObjectName(title + "GrapgicView")
            graphic_view.setMinimumSize(QtCore.QSize(img_w, img_h))
            graphic_view.setMaximumSize(QtCore.QSize(img_w, img_h))
            graphic_view.move(10, 20)
            graphic_view.setScene(scene)
            graphic_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            graphic_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
            pixmap = QPixmap.fromImage(img)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.clear()
            scene.addItem(pixmap_item)
            scene.update()

    def set_callback(self, target:CallbackType, callback):
        self._callback_map[target.name] = callback


    """
    @brief: 设置图像控件的图像
    """
    def set_graphic_widget(self, cv2_img, target:GraphicWidgets):
        # 获取目标graphic view的size
        target_w = self._graphic_widgets[target.name].width()
        target_h = self._graphic_widgets[target.name].height()
        logging.debug("Widget: %s, size: (%d, %d)"%(target.name, target_w, target_h))
        ## 居中铺满放置
        img_w = cv2_img.shape[1]
        img_h = cv2_img.shape[0]
        ## 计算缩放比例 TODO
        #ratio = min(float(target_w)/img_w, float(target_h)/img_h)
        ratio = 1
        img = cv2.resize(cv2_img, (0, 0), fx=ratio, fy=ratio)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y = img.shape[0]
        x = img.shape[1]
        frame = QImage(rgb_img, x, y, x * 3, QImage.Format.Format_RGB888)
        self._update_graphic_signal.emit(frame, target)


    """
    @brief: 在 "图像详情列表" 中绘制以 dict 保存的图像及其名称所构成的列表
    @param:
        - details:
            {
                "title": cv_img
            }
    """
    def set_graphic_detail_list(self, details:dict):
        # 将dict中的cv_img转换为QImage
        for item in details:
            cv_img = details[item]
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            y = cv_img.shape[0]
            x = cv_img.shape[1]
            details[item] = QImage(rgb_img, x, y, x * 3, QImage.Format.Format_RGB888)
        self._set_graphic_detail_signal.emit(details)
