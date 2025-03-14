from QtUI.Ui_TemplateEditor import Ui_TemplateEditor

import os

from enum import Enum
import logging
from types import MethodType
import re
import threading

import cv2

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QRectF, QPointF
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QInputDialog, QMessageBox, QSlider, QSpinBox, QApplication
from PyQt6.QtGui import QPixmap, QImage, QColor, QStandardItemModel, QStandardItem

from QtUI.Widgets.DraggableResizableRect import DraggableResizableRect
from Utils.Config import Config


class TemplateEditorButtonCallbacks(Enum):
    """
    @brief: 按钮事件回调
    """
    OpenTemplatePhotoClicked = 1


class TemplateEditorGraphicViews(Enum):
    """
    @brief: GraphicView控件枚举
    """
    InputGraphicView = 1
    TemplateGraphicView = 2


class ShieldedArea():
    def __init__(self, rect_widget:DraggableResizableRect, id:int) -> None:
        """
        @brief: 屏蔽区对象, 用于引用并记录控件, 并生成dict供录入yaml
        """
        self._widget = rect_widget
        self._id = id

    
    def get_widget(self) -> DraggableResizableRect:
        return self._widget


    def get_id(self) -> int:
        return self._id


class OCR_BarcodePairs():
    def __init__(self) -> None:
        """
        @brief: OCR-条码对照区域
        """
        self._ocr_area = None
        self._barcode_area = None


class _param_changed_source(Enum):
    HMinChagned = 1
    HMaxChanged = 2
    SMinChanged = 3
    SMaxChanged = 4
    DepthThresholdChanged = 5


class EditorUIParams():
    """
    @brief: 参数区运行参数结构体
    @param:
        - h_min: 标签色块寻找时最低色相值
        - h_max: 标签色块寻找时最高色相值
        - s_min: 标签色块寻找时最低饱和度
        - s_max: 标签色块寻找时最高饱和度
        - depth_threshold: 标签的打印样式黑度阈值
    """
    def __init__(self, 
        h_min:int = 0, h_max:int = 170, s_min:int = 13, s_max:int = 255,
        depth_threshold:int = 130
    ) -> None:
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.depth_threshold = depth_threshold


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
        
        # 默认参数
        self._param = EditorUIParams()

        # 参数改变时的回调函数
        self._param_callback = lambda _: logging.warning("params_changed_callback not set.")


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
        self._add_shield_area_signal.connect(self._add_shielded_area, type=Qt.ConnectionType.QueuedConnection)

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

        # 初始化slider及spinbox
        self._bind_slider_spin_to_callback(self.HMinSlider, self.HMinSpinBox, _param_changed_source.HMinChagned)
        self._bind_slider_spin_to_callback(self.HMaxSlider, self.HMaxSpinBox, _param_changed_source.HMaxChanged)
        self._bind_slider_spin_to_callback(self.SMinSlider, self.SMinSpinBox, _param_changed_source.SMinChanged)
        self._bind_slider_spin_to_callback(self.SMaxSlider, self.SMaxSpinBox, _param_changed_source.SMaxChanged)
        self._bind_slider_spin_to_callback(self.DepthThresholdSlider, self.DepthThresholdSpinBox, _param_changed_source.DepthThresholdChanged)

        # 窗口关闭回调函数
        self._save_template_cb = lambda:logging.debug("The callback function for save template is not set.")
        self._window_closed_cb = lambda:logging.debug("The callback function for closing the window is not set.")


    def _bind_slider_spin_to_callback(self, slider:QSlider, spin:QSpinBox, source:_param_changed_source):
        """
        @brief: 绑定slider、spin, 并将其绑定到对应的callback上
        """
        slider.valueChanged.connect(lambda:spin.setValue(slider.value()))
        slider.sliderReleased.connect(lambda source=source, slider=slider, spin=spin:self._param_changed_cb(source, slider, spin))
        spin.valueChanged.connect(lambda source=source, slider=slider, spin=spin:self._param_changed_cb(source, slider, spin))


    @pyqtSlot(_param_changed_source, QSlider, QSpinBox)
    def _param_changed_cb(self, source:_param_changed_source, slider:QSlider, spin:QSpinBox):
        """
        @brief: 参数区变化的槽函数
        """
        # 过滤掉由于Slider改变导致的SpinBox改变引发的信号
        if(
            (QApplication.focusWidget() == spin) or
            (not slider.isSliderDown())
        ):
            # 如果是SpinBox的更改, 则需要同步到Slider
            if(QApplication.focusWidget() == spin):
                slider.setValue(spin.value())
            # 更新数值到params
            match source:
                case _param_changed_source.HMinChagned:
                    self._param.h_min = slider.value()
                case _param_changed_source.HMaxChanged:
                    self._param.h_max = slider.value()
                case _param_changed_source.SMinChanged:
                    self._param.s_min = slider.value()
                case _param_changed_source.SMaxChanged:
                    self._param.s_max = slider.value()
                case _param_changed_source.DepthThresholdChanged:
                    self._param.depth_threshold = self.DepthThresholdSlider.value()
            self._param_callback(self._param)


    @pyqtSlot(int, int, int, int)
    def _add_shielded_area(self, x1:int, y1:int, x2:int, y2:int):
        """
        向模板中增加屏蔽区域方框
        """
        rect = DraggableResizableRect(
            x=x1, 
            y=y1, 
            width=x2-x1, 
            height=y2-y1,
            fill_color=QColor(0, 0, 0, 50),
            edge_color=QColor(0, 0, 0, 100),
            edge_size=5
        )
        # 向scene中添加item
        self._graphic_views_scenes[TemplateEditorGraphicViews.TemplateGraphicView.name].addItem(rect)
        
        # 向ListView中添加元素
        with self._areas_lock:
            row = len(self._shielded_areas_dict)
            self._shielded_area_list_module.setItem(row, 0, QStandardItem(str(x1)))
            self._shielded_area_list_module.setItem(row, 1, QStandardItem(str(y1)))
            self._shielded_area_list_module.setItem(row, 2, QStandardItem(str(x2)))
            self._shielded_area_list_module.setItem(row, 3, QStandardItem(str(y2)))

            # 转为辅助对象
            area = ShieldedArea(rect, self._next_shielded_areas_id)
            # 存入字典
            self._shielded_areas_dict[self._next_shielded_areas_id] = area
            # 配置回调函数
            rect.mouseReleaseEvent = MethodType(self._rect_release_event, area)
            #rect.itemChange = MethodType(self._rect_changed_event, rect)
            # ID自增
            self._next_shielded_areas_id += 1


    @pyqtSlot()
    def _add_shielded_area_callback(self):
        """
        @brief: UI内部专用按钮及逻辑的槽函数
        """
        self._add_shielded_area(x1=0, y1=0, x2=100, y2=50)


    @pyqtSlot()
    def _del_shielded_area_callback(self):
        """
        @brief: 删除屏蔽区域的槽函数
        """
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


    def _btn_callbacks(self, target:TemplateEditorButtonCallbacks):
        """
        @brief: 通用按钮(TemplateEditorButtonCallbacks)点击事件槽函数
        """
        if(not target.name in self._btn_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._btn_callback_map[target.name]()


    def set_btn_callback(self, target:TemplateEditorButtonCallbacks, callback):
        """
        @brief: 设置通用按钮回调函数的接口, 可用于设置 `ButtonCallbackType` 中定义的所有类型回调
        @param:
            - callback: 回调函数, 原型应为 func()
        """
        self._btn_callback_map[target.name] = callback


    @pyqtSlot(QImage, TemplateEditorGraphicViews)
    def _update_graphic_view(self, img:QImage, target:TemplateEditorGraphicViews):
        """
        @brief: 更新图像的槽函数
        """
        pixmap = QPixmap.fromImage(img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self._graphic_views_scenes[target.name].clear()
        self._graphic_views_scenes[target.name].addItem(pixmap_item)
        self._graphic_views_scenes[target.name].update()


    def set_graphic_widget(self, cv2_img, target:TemplateEditorGraphicViews):
        """
        @brief: 设置图像控件的图像
        """
        if(target == TemplateEditorGraphicViews.TemplateGraphicView):
            self._template_shape = cv2_img.shape
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        h = cv2_img.shape[0]
        w = cv2_img.shape[1]
        frame = QImage(rgb_img, w, h, w * 3, QImage.Format.Format_RGB888)
        self._update_graphic_signal.emit(frame, target)


    def set_params_changed_callback(self, callback):
        self._param_callback = callback


    def _rect_changed_event(self, area:ShieldedArea, change, value):
        """
        @brief: 矩形形状改变事件, 用于长度宽度限幅
        """
        DraggableResizableRect.itemChange(area.get_widget(), change, value)


    def _get_coor_from_rect(self, rect:DraggableResizableRect) -> list:
        """
        @brief: 从DraggableResizableRect中获取左上角和右下角坐标
        @return: [x1, y1, x2, y2]
        """
        # 获取左上角右下角坐标
        x1 = rect.scenePos().x()
        y1 = rect.scenePos().y()
        w = rect.rect().width()
        h = rect.rect().height()
        # 修正坐标
        x1 += rect.rect().x()
        y1 += rect.rect().y()

        # 判断左上角是否越界, 同时避免矩形尺寸小于3*3
        if(x1 < 0):
            logging.debug("cond1 x1=%d", x1)
            x1 = 0
        elif(x1 > self._template_shape[1] - w):
            logging.debug("cond2 x1=%d", x1)
            x1 = self._template_shape[1] - w
        if(y1 < 0):
            logging.debug("cond3 y1=%d", y1)
            y1 = 0
        elif(y1 > self._template_shape[0] - h):
            logging.debug("cond4 y1=%d", y1)
            y1 = self._template_shape[0] - h
            
        # 计算右下角坐标
        x2 = x1 + w
        y2 = y1 + h

        # 判断右下角是否越界, 同时避免矩形尺寸小于3*3
        if(x2 < 3):
            logging.debug("cond5 x2=%d", x2)
            x2 = 3
        elif(x2 > self._template_shape[1]):
            logging.debug("cond6 x2=%d", x2)
            x2 = self._template_shape[1]
        if(y2 < 3):
            logging.debug("cond7 y2=%d", y2)
            y2 = 3
        elif(y2 > self._template_shape[0]):
            logging.debug("cond8 y2=%d", y2)
            y2 = self._template_shape[0]
        
        return [x1, y1, x2, y2]


    def _rect_release_event(self, area, event):
        """
        @brief: 鼠标松开事件, 用于边界检测
        """
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
        self._add_shield_area_signal[int, int, int, int].emit(x1, y1, x2, y2)


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
