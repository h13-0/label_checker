from QtUI.Ui_TemplateEditor import Ui_TemplateEditor

from enum import Enum
import logging
from types import MethodType
import threading
import copy

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
    SpecifyBarTenderTemplatePathClicked = 1
    OpenImageSampleClicked              = 2
    AddBarcodeSourceClicked             = 3
    DeleteBarcodeSourceClicked          = 4
    AddOCRSourceClicked                 = 5
    DeleteOCRSourceClicked              = 6


class TemplateEditorGraphicViews(Enum):
    """
    @brief: GraphicView控件枚举
    """
    TenderGraphicsView  = 1
    InputGraphicView    = 2
    SampleGraphicView   = 3
    PatternGraphicsView = 4


class ImageArea():
    def __init__(self, rect_widget:DraggableResizableRect, id:str) -> None:
        """
        @brief: 图像区域对象, 用于绑定ID, 引用并记录控件, 生成dict供录入yaml等
        """
        self._widget = rect_widget
        self._id = id

    
    def get_widget(self) -> DraggableResizableRect:
        return self._widget


    def get_id(self) -> str:
        return self._id


    def __deepcopy__(self, memo):
        """
        @brief: deepcopy的拷贝实现, 注意该类中的_widget控件依旧是共享的, 并非深度拷贝。
        """
        new_area = ImageArea(self._widget, self._id)
        memo[id(self)] = new_area
        return new_area


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
    def __init__(self, 
        h_min:int = 0, h_max:int = 170, s_min:int = 13, s_max:int = 255,
        depth_threshold:int = 130
    ) -> None:
        """
        @brief: 参数区运行参数结构体
        @param:
            - h_min: 标签色块寻找时最低色相值
            - h_max: 标签色块寻找时最高色相值
            - s_min: 标签色块寻找时最低饱和度
            - s_max: 标签色块寻找时最高饱和度
            - depth_threshold: 标签的打印样式黑度阈值
        """
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.depth_threshold = depth_threshold


class TemplateEditorUI(Ui_TemplateEditor, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, TemplateEditorGraphicViews)
    _add_barcode_source_signal = pyqtSignal(str, int, int, int, int)
    _del_barcode_source_signal = pyqtSignal(str)
    def __init__(self, config:Config, name:str = "", template_list:list=[]):
        Ui_TemplateEditor.__init__(self)
        QWidget.__init__(self)

        # 缓存变量
        self._config = config
        self._name = name
        self._template_list = template_list

        ## 按钮回调映射
        self._btn_callback_map = {}

        ## 数据源区域列表
        self._barcode_sources =[]
        self._ocr_sources = []
        self._next_barcode_source_id = 0
        self._next_ocr_source_id = 0
        self._sources_lock = threading.Lock()
        
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
        self._barcode_source_list_module = QStandardItemModel(self.BarcodeSourcesList)
        self._barcode_source_list_module.setHorizontalHeaderItem(0, QStandardItem("数据源ID"))
        self._barcode_source_list_module.setHorizontalHeaderItem(1, QStandardItem("x1"))
        self._barcode_source_list_module.setHorizontalHeaderItem(2, QStandardItem("y1"))
        self._barcode_source_list_module.setHorizontalHeaderItem(3, QStandardItem("x2"))
        self._barcode_source_list_module.setHorizontalHeaderItem(4, QStandardItem("y2"))

        self.BarcodeSourcesList.setModel(self._barcode_source_list_module)
        self.BarcodeSourcesList.setColumnWidth(0, 75)
        self.BarcodeSourcesList.setColumnWidth(1, 75)
        self.BarcodeSourcesList.setColumnWidth(2, 75)
        self.BarcodeSourcesList.setColumnWidth(3, 75)
        self.BarcodeSourcesList.setColumnWidth(4, 75)

        self._ocr_barcode_list_module = QStandardItemModel(self.OCR_BarcodePairsList)
        self._ocr_barcode_list_module.setHorizontalHeaderItem(0, QStandardItem("Barcode x1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(1, QStandardItem("Barcode y1"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(2, QStandardItem("Barcode x2"))
        self._ocr_barcode_list_module.setHorizontalHeaderItem(3, QStandardItem("Barcode y2"))
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
        self.SpecifyBarTenderTemplatePath.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.SpecifyBarTenderTemplatePathClicked))
        self.OpenImageSample.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.OpenImageSampleClicked))
        self.AddBarcodeSource.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.AddBarcodeSourceClicked))
        self.DeleteBarcodeSource.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.DeleteBarcodeSourceClicked))
        self.AddOCRSource.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.AddBarcodeSourceClicked))
        self.DeleteOCRSource.clicked.connect(lambda:self._btn_callbacks(TemplateEditorButtonCallbacks.DeleteBarcodeSourceClicked))

        ## 连接自定义信号
        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._add_barcode_source_signal.connect(self._add_barcode_source_area, type=Qt.ConnectionType.QueuedConnection)
        self._del_barcode_source_signal.connect(self._del_barcode_source_slot, type=Qt.ConnectionType.QueuedConnection)

        # 初始化GraphicView映射
        self._graphic_views = {
            "TenderGraphicsView": self.TenderGraphicsView,
            "InputGraphicView": self.InputGraphicView,
            "SampleGraphicView": self.SampleGraphicView,
            "PatternGraphicsView": self.PatternGraphicsView,
        }
        self._graphic_views_scenes = {
            "TenderGraphicsView": QGraphicsScene(),
            "InputGraphicView": QGraphicsScene(),
            "SampleGraphicView": QGraphicsScene(),
            "PatternGraphicsView": QGraphicsScene(),
        }
        for widget in self._graphic_views:
            self._graphic_views[widget].setScene(
                self._graphic_views_scenes[widget]
            )
            self._graphic_views[widget].wheelEvent = MethodType(self._wheel_event, self._graphic_views[widget])
            self._graphic_views[widget].setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        
        # 若干图像的大小尺寸
        self._bartender_img_shape = None
        self._img_sample_shape = None

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


    @pyqtSlot(str, int, int, int, int)
    def _add_barcode_source_area(self, id:str, x1:int, y1:int, x2:int, y2:int):
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
        self._graphic_views_scenes[TemplateEditorGraphicViews.TenderGraphicsView.name].addItem(rect)
        
        # 向ListView中添加元素
        with self._sources_lock:
            row = len(self._barcode_sources)
            self._barcode_source_list_module.setItem(row, 0, QStandardItem(id))
            self._barcode_source_list_module.setItem(row, 1, QStandardItem(str(x1)))
            self._barcode_source_list_module.setItem(row, 2, QStandardItem(str(y1)))
            self._barcode_source_list_module.setItem(row, 3, QStandardItem(str(x2)))
            self._barcode_source_list_module.setItem(row, 4, QStandardItem(str(y2)))

            # 转为辅助对象
            area = ImageArea(rect, id)
            # 存入数据源数组
            self._barcode_sources.append(area)
            # 配置回调函数
            rect.mouseReleaseEvent = MethodType(self._rect_release_event, area)
            #rect.itemChange = MethodType(self._rect_changed_event, rect)


    @pyqtSlot(str)
    def _del_barcode_source_slot(self, id:str):
        """
        @brief: 删除条码源的槽函数
        """
        # 从后向前遍历表格中所有行
        for row in range(self._barcode_source_list_module.rowCount() - 1, -1, -1):
            # 获取指定行和列的项
            item = self._barcode_source_list_module.item(row, 0)
            # 检查该项的值是否与给定值相匹配
            if item and item.text() == id:
                # 如果匹配，删除整行
                self._barcode_source_list_module.removeRow(row)

        for row in range(self._ocr_barcode_list_module.rowCount() - 1, -1, -1):
            # 获取指定行和列的项
            item = self._ocr_barcode_list_module.item(row, 0)
            # 检查该项的值是否与给定值相匹配
            if item and item.text() == id:
                # 如果匹配，删除整行
                self._ocr_barcode_list_module.removeRow(row)

        # 删除list中数据
        with self._sources_lock:
            for i in range(len(self._barcode_sources) - 1, -1, -1):
                if(self._barcode_sources[i].get_id() == id):
                    self._graphic_views_scenes[TemplateEditorGraphicViews.TenderGraphicsView.name].removeItem(self._barcode_sources[i].get_widget())
                    self._barcode_sources.pop(i)
            for i in range(len(self._ocr_sources) - 1, -1, -1):
                if(self._ocr_sources[i].get_id() == id):
                    self._graphic_views_scenes[TemplateEditorGraphicViews.TenderGraphicsView.name].removeItem(self._ocr_sources[i].get_widget())
                    self._ocr_sources.pop(i)


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
        if(target == TemplateEditorGraphicViews.SampleGraphicView):
            self._img_sample_shape = cv2_img.shape
        elif(target == TemplateEditorGraphicViews.TenderGraphicsView):
            self._bartender_img_shape = cv2_img.shape
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        h = cv2_img.shape[0]
        w = cv2_img.shape[1]
        frame = QImage(rgb_img, w, h, w * 3, QImage.Format.Format_RGB888)
        self._update_graphic_signal.emit(frame, target)


    def set_params_changed_callback(self, callback):
        self._param_callback = callback


    def _rect_changed_event(self, area:ImageArea, change, value):
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
        elif(x1 > self._bartender_img_shape[1] - w):
            logging.debug("cond2 x1=%d", x1)
            x1 = self._bartender_img_shape[1] - w
        if(y1 < 0):
            logging.debug("cond3 y1=%d", y1)
            y1 = 0
        elif(y1 > self._bartender_img_shape[0] - h):
            logging.debug("cond4 y1=%d", y1)
            y1 = self._bartender_img_shape[0] - h
            
        # 计算右下角坐标
        x2 = x1 + w
        y2 = y1 + h

        # 判断右下角是否越界, 同时避免矩形尺寸小于3*3
        if(x2 < 3):
            logging.debug("cond5 x2=%d", x2)
            x2 = 3
        elif(x2 > self._bartender_img_shape[1]):
            logging.debug("cond6 x2=%d", x2)
            x2 = self._bartender_img_shape[1]
        if(y2 < 3):
            logging.debug("cond7 y2=%d", y2)
            y2 = 3
        elif(y2 > self._bartender_img_shape[0]):
            logging.debug("cond8 y2=%d", y2)
            y2 = self._bartender_img_shape[0]
        
        return [x1, y1, x2, y2]


    def _rect_release_event(self, area, event):
        """
        @brief: 鼠标松开事件, 用于边界检测
        """
        rect = None
        if(isinstance(area, ImageArea)):
            rect = area.get_widget()
        else:
            # todo
            pass
        logging.debug("rect: " + str(rect.rect()))
        logging.debug("pos: " + str(rect.scenePos()))
        logging.debug("img shape: " + str(self._bartender_img_shape))
        DraggableResizableRect.mouseReleaseEvent(rect, event)

        x1, y1, x2, y2 = self._get_coor_from_rect(rect)
        w = x2 - x1
        h = y2 - y1

        rect.setPos(QPointF(x1, y1))
        rect.setRect(QRectF(0, 0, w, h))
        
        # 更新数据到ListView
        barcode_sources = []
        ocr_sources = []
        with self._sources_lock:
            barcode_sources = copy.deepcopy(self._barcode_sources)
            ocr_sources = copy.deepcopy(self._ocr_sources)
        if(area in barcode_sources):
            ## 以下操作需要Python 3.7+
            row = barcode_sources.index(barcode_sources.get_id())
            self._barcode_source_list_module.setItem(row, 1, QStandardItem(str(round(x1))))
            self._barcode_source_list_module.setItem(row, 2, QStandardItem(str(round(y1))))
            self._barcode_source_list_module.setItem(row, 3, QStandardItem(str(round(x2))))
            self._barcode_source_list_module.setItem(row, 4, QStandardItem(str(round(y2))))
        elif(area in ocr_sources):
            # todo
            pass


    def _window_close_event(self, window, event):
        dialog = QInputDialog()
        dialog.setWindowTitle("保存模板配置")
        dialog.setLabelText("模板名称：")
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
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


    def add_barcode_source(self, id:str, x1:int, y1:int, x2:int, y2:int):
        """
        @brief: 增加条码数据源
        @param:
            - id: 数据源ID名称
            - x1: 左上角x轴坐标
            - y1: 左上角y轴坐标
            - x2: 右下角x轴坐标
            - y2: 右下角y轴坐标
        """
        self._add_barcode_source_signal.emit(id, x1, y1, x2, y2)


    def get_barcode_sources(self) -> list:
        """
        @brief: 获取条码源列表
        """
        barcode_sources = []
        with self._sources_lock:
            barcode_sources = copy.deepcopy(self._barcode_sources)
        return barcode_sources


    def get_selected_barcode_sources(self) -> list:
        """
        @brief: 获取当前在列表中选中的条码数据源
        @return:
            由id(str)组成的list
        """
        # TODO: 确保在主线程中读取UI控件
        selected_indexs = self.BarcodeSourcesList.selectionModel().selectedIndexes()
        selected_ids = []

        for index in selected_indexs:
            selected_ids.append(self._barcode_source_list_module.item(index.row(), 0).text())

        return selected_ids


    def del_barcode_source(self, target_id:str):
        self._del_barcode_source_signal.emit(target_id)

    def get_ocr_sources(self) -> list:
        """
        @brief: 获取OCR源列表
        """
        ocr_sources = []
        with self._sources_lock:
            ocr_sources = copy.deepcopy(self._ocr_sources)
        return ocr_sources


    def add_template_to_table(self, name:str):
        pass


    def delete_template_by_name(self, name:str):
        pass

    def set_delete_template_callback(self, callback):
        pass


    def set_window_closed_callback(self, callback):
        self._window_closed_cb = callback


    def set_save_template_callback(self, callback):
        self._save_template_cb = callback


