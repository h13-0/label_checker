from QtUI.Ui_TemplateEditor import Ui_TemplateEditor

from enum import Enum
import logging
from types import MethodType

import cv2

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView
from PyQt6.QtGui import QPixmap, QImage

"""
@brief: 按钮事件回调
"""
class TemplateEditorButtonCallbacks(Enum):
    OpenTemplatePhotoClicked = 1


"""
@brief: GraphicView控件枚举
"""
class TemplateEditorGraphicViews(Enum):
    TemplateGraphicView = 1


class TemplateEditorUI(Ui_TemplateEditor, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, TemplateEditorGraphicViews)
    def __init__(self):
        Ui_TemplateEditor.__init__(self)
        QWidget.__init__(self)

        ## 按钮回调映射
        self._btn_callback_map = {}


    def _wheel_event(self, widget, event):
        if(True):
            delta = event.angleDelta().y()
            scale = 1 + delta / 1000.0

            if(
                widget == self.MainGraphicView or
                widget == self.TemplateGraphicView
            ):
                widget.scale(scale, scale)



    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        # 连接信号槽
        ## 连接按钮信号
        self.OpenTemplatePhoto.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.OpenTemplatePhotoClicked))

        ## 连接自定义信号
        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)

        # 初始化GraphicView映射
        self._graphic_views = {
            "TemplateGraphicView": self.TemplateGraphicView,
        }
        self._graphic_views_scenes = {
            "TemplateGraphicView": QGraphicsScene()
        }
        for widget in self._graphic_views:
            self._graphic_views[widget].setScene(
                self._graphic_views_scenes[widget]
            )
            self._graphic_views[widget].wheelEvent = MethodType(self._wheel_event, self._graphic_views[widget])
            self._graphic_views[widget].setDragMode(QGraphicsView.DragMode.ScrollHandDrag)


    """
    @brief: 按钮点击事件槽函数
    """
    def _btn_callbacks(self, target:TemplateEditorButtonCallbacks):
        if(not target.name in self._btn_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._btn_callback_map[target.name]()


    """
    @brief: 设置按钮回调函数的接口, 可用于设置 `ButtonCallbackType` 中定义的所有类型回调
    @param:
        - callback: 回调函数, 原型应为 func()
    """
    def set_btn_callback(self, target:TemplateEditorButtonCallbacks, callback):
        self._btn_callback_map[target.name] = callback


    """
    @brief: 更新图像的槽函数
    """
    def _update_graphic_view(self, img:QImage, target:TemplateEditorGraphicViews):
        pixmap = QPixmap.fromImage(img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self._graphic_views_scenes[target.name].clear()
        self._graphic_views_scenes[target.name].addItem(pixmap_item)
        self._graphic_views_scenes[target.name].update()


    """
    @brief: 设置图像控件的图像
    """
    def set_graphic_widget(self, cv2_img, target:TemplateEditorGraphicViews):
        # 获取目标graphic view的size
        target_w = self._graphic_views[target.name].width()
        target_h = self._graphic_views[target.name].height()
        logging.debug("Widget: %s, size: (%d, %d)"%(target.name, target_w, target_h))
        ## 居中铺满放置
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        y = cv2_img.shape[0]
        x = cv2_img.shape[1]
        frame = QImage(rgb_img, x, y, x * 3, QImage.Format.Format_RGB888)
        self._update_graphic_signal.emit(frame, target)
