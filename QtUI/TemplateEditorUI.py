from QtUI.Ui_TemplateEditor import Ui_TemplateEditor

import os

from enum import Enum
import logging
from types import MethodType
import re
import threading

import cv2

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QRectF, QPointF
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QInputDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QImage, QColor, QStandardItemModel, QStandardItem

from QtUI.Widgets.DraggableResizableRect import DraggableResizableRect
from Template.Template import Template
from Utils.Config import Config


"""
@brief: 按钮事件回调
"""
class TemplateEditorButtonCallbacks(Enum):
    OpenTemplatePhotoClicked = 1


"""
@brief: GraphicView控件枚举
"""
class TemplateEditorGraphicViews(Enum):
    InputGraphicView = 1
    TemplateGraphicView = 2


class ShieldedArea():
    def __init__(self, rect_widget:DraggableResizableRect, id:int) -> None:
        """屏蔽区对象, 用于引用并记录控件, 并生成dict供录入yaml"""
        self._widget = rect_widget
        self._id = id

    
    def get_widget(self) -> DraggableResizableRect:
        return self._widget

    def get_id(self) -> int:
        return self._id


"""
@brief: OCR-条码对照区域
"""
class OCR_BarcodePairs():
    def __init__(self) -> None:
        self._ocr_area = None
        self._barcode_area = None


class TemplateEditorUI(Ui_TemplateEditor, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, TemplateEditorGraphicViews)
    _add_shield_area_signal = pyqtSignal(int, int, int, int)
    def __init__(self, config:Config, name:str = "", template_list:list=[]):
        Ui_TemplateEditor.__init__(self)
        QWidget.__init__(self)

        # 缓存变量
        self._config = config
        self._name = name
        self._template_list = template_list

        ## 按钮回调映射
        self._btn_callback_map = {}

        ## 屏蔽区域及对照区域列表
        self._shielded_areas_dict = {}
        self._ocr_barcode_comparison_pairs_dict = {}
        self._next_shielded_areas_id = 0
        self._next_ocr_barcode_comparison_pairs_id = 0
        self._areas_lock = threading.Lock()
        

    def _wheel_event(self, widget, event):
        """
        GraphicView的滚轮事件响应
        """
        delta = event.angleDelta().y()
        scale = 1 + delta / 1000.0
        widget.scale(scale, scale)


    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        # 绑定MainWindow关闭事件
        MainWindow.closeEvent = MethodType(self._window_close_event, MainWindow)

        # 初始化UI
        ## 初始化tableView
        self._shielded_area_list_module = QStandardItemModel(self.ShieldedAreaList)
        self._shielded_area_list_module.setHorizontalHeaderItem(0, QStandardItem("x1"))
        self._shielded_area_list_module.setHorizontalHeaderItem(1, QStandardItem("y1"))
        self._shielded_area_list_module.setHorizontalHeaderItem(2, QStandardItem("x2"))
        self._shielded_area_list_module.setHorizontalHeaderItem(3, QStandardItem("y2"))
        self.ShieldedAreaList.setModel(self._shielded_area_list_module)
        self.ShieldedAreaList.setColumnWidth(0, 75)
        self.ShieldedAreaList.setColumnWidth(1, 75)
        self.ShieldedAreaList.setColumnWidth(2, 75)
        self.ShieldedAreaList.setColumnWidth(3, 75)

        self._ocr_barcode_list_module = QStandardItemModel(self.OCR_BarcodePairsList)
        self._ocr_barcode_list_module.setHorizontalHeaderItem(0, QStandardItem("Barcode x1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(1, QStandardItem("Barcode  y1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(2, QStandardItem("Barcode  x2"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(3, QStandardItem("Barcode  y2"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(4, QStandardItem("OCR x1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(5, QStandardItem("OCR y1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(6, QStandardItem("OCR x2"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(7, QStandardItem("OCR y2"))
        self.OCR_BarcodePairsList.setModel(self._ocr_barcode_list_module)
        self.OCR_BarcodePairsList.resizeColumnToContents(0)
        self.OCR_BarcodePairsList.resizeColumnToContents(1)
        self.OCR_BarcodePairsList.resizeColumnToContents(2)
        self.OCR_BarcodePairsList.resizeColumnToContents(3)
        self.OCR_BarcodePairsList.resizeColumnToContents(4)
        self.OCR_BarcodePairsList.resizeColumnToContents(5)
        self.OCR_BarcodePairsList.resizeColumnToContents(6)
        self.OCR_BarcodePairsList.resizeColumnToContents(7)


        # 连接信号槽
        ## 连接按钮信号
        ### 专用按钮
        self.AddShieldedArea.clicked.connect(self._add_shielded_area_callback)
        self.DeleteShieldedArea.clicked.connect(self._del_shielded_area_callback)

        ### 通用按钮
        self.OpenTemplatePhoto.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.OpenTemplatePhotoClicked))

        ## 连接自定义信号
        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._add_shield_area_signal.connect(self._add_shielded_area)

        # 初始化GraphicView映射
        self._graphic_views = {
            "InputGraphicView": self.InputGraphicView,
            "TemplateGraphicView": self.TemplateGraphicView,
        }
        self._graphic_views_scenes = {
            "InputGraphicView": QGraphicsScene(),
            "TemplateGraphicView": QGraphicsScene(),
        }
        for widget in self._graphic_views:
            self._graphic_views[widget].setScene(
                self._graphic_views_scenes[widget]
            )
            self._graphic_views[widget].wheelEvent = MethodType(self._wheel_event, self._graphic_views[widget])
            self._graphic_views[widget].setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        self._template_shape = None

        # 窗口关闭回调函数
        self._save_template_cb = lambda:logging.debug("The callback function for save template is not set.")
        self._window_closed_cb = lambda:logging.debug("The callback function for closing the window is not set.")


    @pyqtSlot()
    def _add_shielded_area(self, x1:int, y1:int, x2:int, y2:int):
        """
        向模板中增加屏蔽区域方框
        """
        logging.debug("receive")
        rect = DraggableResizableRect(
            x=x1, 
            y=y1, 
            width=x2-x1, 
            height=y2-y1,
            fill_color=QColor(0, 0, 0, 10),
            edge_color=QColor(0, 0, 0, 100),
            edge_size=5
        )
        # 向scene中添加item
        self._graphic_views_scenes[TemplateEditorGraphicViews.TemplateGraphicView.name].addItem(rect)
        
        # 向ListView中添加元素
        with self._areas_lock:
            row = len(self._shielded_areas_dict)
            self._shielded_area_list_module.setItem(row, 0, QStandardItem(str(0)))
            self._shielded_area_list_module.setItem(row, 1, QStandardItem(str(0)))
            self._shielded_area_list_module.setItem(row, 2, QStandardItem(str(100)))
            self._shielded_area_list_module.setItem(row, 3, QStandardItem(str(100)))

            # 转为辅助对象
            area = ShieldedArea(rect, self._next_shielded_areas_id)
            # 存入字典
            self._shielded_areas_dict[self._next_shielded_areas_id] = area
            # 配置回调函数
            rect.mouseReleaseEvent = MethodType(self._rect_release_event, area)
            #rect.itemChange = MethodType(self._rect_changed_event, rect)
            # ID自增
            self._next_shielded_areas_id += 1


    """
    @brief: UI内部专用按钮及逻辑的槽函数
    """
    @pyqtSlot()
    def _add_shielded_area_callback(self):
        self._add_shielded_area(x1=0, y1=0, x2=100, y2=50)


    @pyqtSlot()
    def _del_shielded_area_callback(self):
        """删除屏蔽区域的槽函数"""
        selected_indexs = self.ShieldedAreaList.selectionModel().selectedIndexes()
        selected_rows = []
        # 将所选中的行倒序插入列表, 方便删除
        for index in selected_indexs:
            if(not index.row() in selected_rows):
                selected_rows.insert(0, index.row())
        with self._areas_lock:
        # 删除对应元素及列表
            for row in selected_rows:
                ## 从scene中删除对应元素
                key = list(self._shielded_areas_dict.keys())[row]
                self._graphic_views_scenes[TemplateEditorGraphicViews.TemplateGraphicView.name].removeItem(self._shielded_areas_dict[key].get_widget())
                ## 从列表中删除对应元素
                self._shielded_area_list_module.removeRow(row)
                ## 从dict中删除对应元素
                self._shielded_areas_dict.pop(key)


    """
    @brief: 通用按钮(TemplateEditorButtonCallbacks)点击事件槽函数
    """
    def _btn_callbacks(self, target:TemplateEditorButtonCallbacks):
        if(not target.name in self._btn_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._btn_callback_map[target.name]()


    """
    @brief: 设置通用按钮回调函数的接口, 可用于设置 `ButtonCallbackType` 中定义的所有类型回调
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
        if(target == TemplateEditorGraphicViews.TemplateGraphicView):
            self._template_shape = cv2_img.shape
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        h = cv2_img.shape[0]
        w = cv2_img.shape[1]
        frame = QImage(rgb_img, w, h, w * 3, QImage.Format.Format_RGB888)
        self._update_graphic_signal.emit(frame, target)


    def _rect_changed_event(self, area:ShieldedArea, change, value):
        """矩形形状改变事件, 用于长度宽度限幅"""

        DraggableResizableRect.itemChange(area.get_widget(), change, value)


    def _get_coor_from_rect(self, rect:DraggableResizableRect) -> list:
        """
        从DraggableResizableRect中获取左上角和右下角坐标

        Return: [x1, y1, x2, y2]
        """
        # 获取左上角右下角坐标
        x1 = rect.scenePos().x()
        y1 = rect.scenePos().y()
        w = rect.rect().width()
        h = rect.rect().height()
        # 修正坐标
        x1 -= rect.rect().x()
        y1 -= rect.rect().y()

        # 判断左上角是否越界, 同时避免矩形尺寸小于3*3
        if(x1 < 0):
            x1 = 0
        elif(x1 > self._template_shape[1] - w):
            x1 = self._template_shape[1] - w
        if(y1 < 0):
            y1 = 0
        elif(y1 > self._template_shape[0] - h):
            y1 = self._template_shape[0] - h
            
        # 计算右下角坐标
        x2 = x1 + w
        y2 = y1 + h

        # 判断右下角是否越界, 同时避免矩形尺寸小于3*3
        if(x2 < 3):
            x2 = 3
        elif(x2 > self._template_shape[1]):
            x2 = self._template_shape[1]
        if(y2 < 3):
            y2 = 3
        elif(y2 > self._template_shape[0]):
            y2 = self._template_shape[0]
        
        return [x1, y1, x2, y2]


    def _rect_release_event(self, area, event):
        """鼠标松开事件, 用于边界检测"""
        rect = None
        if(isinstance(area, ShieldedArea)):
            rect = area.get_widget()
        else:
            # todo
            pass
        logging.debug("rect: " + str(rect.rect()))
        logging.debug("pos: " + str(rect.scenePos()))
        logging.debug("img shape: " + str(self._template_shape))
        DraggableResizableRect.mouseReleaseEvent(rect, event)

        x1, y1, x2, y2 = self._get_coor_from_rect(rect)
        w = x2 - x1
        h = y2 - y1

        rect.setPos(QPointF(x1, y1))
        rect.setRect(QRectF(0, 0, w, h))
        
        # 更新数据到ListView
        if(isinstance(area, ShieldedArea)):
            ## 以下操作需要Python 3.7+
            shielded_areas_list = []
            with self._areas_lock:
                shielded_areas_list = list(self._shielded_areas_dict.keys())
            row = shielded_areas_list.index(area.get_id())
            self._shielded_area_list_module.setItem(row, 0, QStandardItem(str(round(x1))))
            self._shielded_area_list_module.setItem(row, 1, QStandardItem(str(round(y1))))
            self._shielded_area_list_module.setItem(row, 2, QStandardItem(str(round(x2))))
            self._shielded_area_list_module.setItem(row, 3, QStandardItem(str(round(y2))))
        else:
            # todo
            pass


    def _window_close_event(self, window, event):
        dialog = QInputDialog()
        dialog.setWindowTitle("保存模板配置")
        dialog.setLabelText("模板名称：")
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        #dialog.findChild(QLineEdit)
        dialog.setTextValue(self._name)

        ok_pressed = dialog.exec()
        if(ok_pressed): 
            name = dialog.textValue()
            #if(self._save_template(name)):
            if(self._save_template_cb(name)):
                self._window_closed_cb()
                event.accept()
            else:
                event.ignore()
        else:
            if(QMessageBox.question(self, "警告", "确定放弃保存?") == QMessageBox.StandardButton.Yes):
                self._window_closed_cb()
                event.accept()
            else:
                event.ignore()


    def add_shielded_area(self, x1:int, y1:int, x2:int, y2:int):
        logging.debug("add")
        self._add_shield_area_signal.emit(x1, y1, x2, y2)


    def set_window_closed_callback(self, callback):
        self._window_closed_cb = callback


    def set_save_template_callback(self, callback):
        self._save_template_cb = callback


    def get_shield_areas(self) -> list:
        areas = []
        ## 导出屏蔽区域列表
        with self._areas_lock:
            for key in self._shielded_areas_dict:
                area = self._shielded_areas_dict[key]
                areas.append(self._get_coor_from_rect(area.get_widget()))
        return areas
