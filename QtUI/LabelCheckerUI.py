from QtUI.Ui_LabelChecker import Ui_LabelChecker
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QMessageBox, QApplication, QMainWindow
from PyQt6.QtGui import QPixmap, QImage
import logging
from enum import Enum
import cv2
from functools import partial
from types import MethodType

"""
@brief: 按钮事件回调
"""
class ButtonCallbackType(Enum):
    EditTemplateButton = 1
    OpenTargetStreamClicked = 2
    OpenTargetPhotoClicked = 3
    LoadParamsClicked = 4


"""
@brief: ComboBox所选目标改变事件的回调
"""
class ComboBoxChangedCallback(Enum):
    TemplatesChanged = 1


"""
@brief: 窗口的常驻图像控件(GraphicView)的枚举类
"""
class GraphicWidgets(Enum):
    MainGraphicView = 1
    TemplateGraphicView = 2


"""
@brief: 进度条枚举类
"""
class ProgressBarWidgts(Enum):
    CompareProgressBar = 1


"""
@brief: 参数区运行参数结构体
@param:
    - h_min: 标签色块寻找时最低色相值
    - h_max: 标签色块寻找时最高色相值
    - s_min: 标签色块寻找时最低饱和度
    - s_max: 标签色块寻找时最高饱和度
    - depth_threshold: 标签的打印样式黑度阈值
    - class_similarity: 判定同类标签的相似度
    - not_good_similarity: 同类标签中不合格标签的相似度
    - linear_error: 容许的线性误差
    - defect_min_area: 检出缺陷最小面积
"""
class WorkingParams():
    def __init__(self, 
        h_min:int = 0, h_max:int = 170, s_min:int = 13, s_max:int = 255,
        depth_threshold:int = 130, class_similarity:int = 90, 
        not_good_similarity:int = 95, linear_error:int = 7,
        defect_min_area:int = 6
    ) -> None:
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.depth_threshold = depth_threshold
        self.class_similarity = class_similarity
        self.not_good_similarity = not_good_similarity
        self.linear_error = linear_error
        self.defect_min_area = defect_min_area


"""
@brief: 私有类, 用于标识参数变化源
"""
class _param_changed_source(Enum):
    HMinChagned = 1
    HMaxChanged = 2
    SMinChanged = 3
    SMaxChanged = 4
    DepthThresholdChanged = 5
    ClassSimilarityChanged = 6
    NotGoodSimilarityChanged = 7
    LinearErrorChanged = 8
    DefectMinAreaChanged = 9


class LabelCheckerUI(Ui_LabelChecker, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, GraphicWidgets)
    _set_graphic_detail_signal = pyqtSignal(dict)
    _make_msg_box_signal = pyqtSignal(str, str)
    _add_item_to_templates_combo_box_signal = pyqtSignal(str)
    _clear_combo_box_signal = pyqtSignal()
    ## 进度条操作信号
    _set_progress_bar_value_signal = pyqtSignal(ProgressBarWidgts, int)

    def __init__(self):
        Ui_LabelChecker.__init__(self)
        QWidget.__init__(self)

        # 按钮事件回调映射
        self._btn_callback_map = {}
        # ComboBox改变回调映射
        self._cb_changed_callback_map = {}
        # 参数改变的回调函数
        self._param_callback = lambda: logging.warning("params_changed_callback not set.")

        # 默认参数
        self._param = WorkingParams()



    def _wheel_event(self, widget, event):
        if(True):
            delta = event.angleDelta().y()
            scale = 1 + delta / 1000.0

            if(
                widget == self.MainGraphicView or
                widget == self.TemplateGraphicView
            ):
                widget.scale(scale, scale)
            elif(widget in self._graphic_details_graphic_view_list):
                for i in range(len(self._graphic_details_graphic_view_list)):
                    self._graphic_details_graphic_view_list[i].scale(scale, scale)
                    width = self._graphic_details_graphic_view_list[i].width() + 20
                    height = self._graphic_details_graphic_view_list[i].height() + 30
                    self._graphic_details_group_box_list[i].setMaximumWidth(width)
                    self._graphic_details_group_box_list[i].setMinimumWidth(width)
                    self._graphic_details_group_box_list[i].setMaximumHeight(height)
                    self._graphic_details_group_box_list[i].setMinimumHeight(height)


    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        # 进度条控件映射
        self._progress_bars = {
            "CompareProgressBar": self.CopareProgressBar
        }

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
            self._graphic_widgets[widgets].setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        # "图像详情列表"的GroupBox、GraphicView及GraphicsScene
        self._graphic_details_group_box_list = []
        self._graphic_details_graphic_view_list = []
        self._graphic_details_graphic_scene_list = []

        # 连接回调函数
        ## 模板列表(ComboBox)选中事件
        self.TemplatesComboBox.currentIndexChanged.connect(lambda:self._cb_changed_callbacks(ComboBoxChangedCallback.TemplatesChanged))

        ## 按钮点击事件
        self.EditTemplateButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.EditTemplateButton))
        self.OpenTargetStreamButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.OpenTargetStreamClicked))
        self.OpenTargetPhotoButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.OpenTargetPhotoClicked))

        ## 参数区回调函数
        ### 创建参数枚举和参数控件的映射
        self._param_widgets = {
            _param_changed_source.ClassSimilarityChanged: [self.ClassSimilaritySldider, self.ClassSimilaritySpinBox],
            _param_changed_source.NotGoodSimilarityChanged: [self.NotGoodSimilaritySlider, self.NotGoodSimilaritySpinBox],
            _param_changed_source.LinearErrorChanged: [self.LinearErrorSlider, self.LinearErrorSpinBox],
            _param_changed_source.DefectMinAreaChanged: [self.DefectMinAreaSlider, self.DefectMinAreaSpinBox],
        }
        self._connect_param_widgets_signal(self._param_widgets)

        ### 滑动条滑动时, 将值实时更新到SpinBox, TODO: 并入self._connect_param_widgets_signal
        self.ClassSimilaritySldider.valueChanged.connect(lambda:self.ClassSimilaritySpinBox.setValue(self.ClassSimilaritySldider.value()))
        self.NotGoodSimilaritySlider.valueChanged.connect(lambda:self.NotGoodSimilaritySpinBox.setValue(self.NotGoodSimilaritySlider.value()))
        self.LinearErrorSlider.valueChanged.connect(lambda:self.LinearErrorSpinBox.setValue(self.LinearErrorSlider.value()))
        self.DefectMinAreaSlider.valueChanged.connect(lambda:self.DefectMinAreaSpinBox.setValue(self.DefectMinAreaSlider.value()))
        
        ## GraphicView 鼠标滚动缩放
        self.MainGraphicView.wheelEvent = MethodType(self._wheel_event, self.MainGraphicView)
        self.TemplateGraphicView.wheelEvent = MethodType(self._wheel_event, self.TemplateGraphicView)

        ## 连接自定义信号
        ### 向ComboBox添加元素
        self._add_item_to_templates_combo_box_signal.connect(self._add_item_to_templates_combo_box)
        ### 清除ComboBox中所有元素
        self._clear_combo_box_signal.connect(self._clear_combo_box)
        ### GraphicView 更新图像
        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._set_graphic_detail_signal.connect(self._set_graphic_detail_content, type=Qt.ConnectionType.BlockingQueuedConnection)
        ### MessageBox
        self._make_msg_box_signal.connect(self._make_msg_box)
        ### 进度条
        self._set_progress_bar_value_signal.connect(self._set_progress_bar_value)


    def _connect_param_widgets_signal(self, param_widget_map:dict):
        # 将Slider释放信号和SpinBox修改的信号连接到 `_param_changed_cb`
        for param_enum in param_widget_map:
            slider = param_widget_map[param_enum][0]
            spin_box = param_widget_map[param_enum][1]
            # 将Slider释放信号连接到 `_param_changed_cb`
            slider.sliderReleased.connect(partial(self._param_changed_cb, param_enum))
            # 将SpinBox修改的信号连接到 `_param_changed_cb`
            spin_box.valueChanged.connect(partial(self._param_changed_cb, param_enum))
            # 当Slider滑动时, 将值实时更新到对应的SpinBox
            ## TODO


    """
    @brief: ComboBox所选目标改变的回调函数
    """
    def _cb_changed_callbacks(self, target:ComboBoxChangedCallback):
        if(not target.name in self._cb_changed_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._cb_changed_callback_map[target.name](self.TemplatesComboBox.currentIndex(), self.TemplatesComboBox.currentText())


    """
    @brief: 按钮的槽函数, 回调事件转发器
    """
    def _btn_callbacks(self, target:ButtonCallbackType):
        if(not target.name in self._btn_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._btn_callback_map[target.name]()


    """
    @brief: 设置进度条进度的槽函数
    """
    def _set_progress_bar_value(self, target:ProgressBarWidgts, value:int):
        self._progress_bars[target.name].setValue(value)
        logging.debug(value)


    """
    @brief: 更新图像的槽函数
    """
    def _update_graphic_view(self, img:QImage, target:GraphicWidgets):
        pixmap = QPixmap.fromImage(img)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self._graphic_widgets_scenes[target.name].clear()
        self._graphic_widgets_scenes[target.name].addItem(pixmap_item)
        self._graphic_widgets_scenes[target.name].update()


    """
    @brief: 参数区变化的槽函数
    """
    def _param_changed_cb(self, source:_param_changed_source):
        # 获取事件对应的Slider和SpinBox
        slider = self._param_widgets[source][0]
        spin_box = self._param_widgets[source][1]

        # 过滤掉由于Slider改变导致的SpinBox改变引发的信号
        if(
            (QApplication.focusWidget() == spin_box) or
            (not slider.isSliderDown())
        ):
            # 如果是SpinBox的更改, 则需要同步到Slider
            if(QApplication.focusWidget() == spin_box):
                slider.setValue(spin_box.value())
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
                    self._param.depth_threshold = slider.value()
                case _param_changed_source.ClassSimilarityChanged:
                    self._param.class_similarity = slider.value()
                case _param_changed_source.NotGoodSimilarityChanged:
                    self._param.not_good_similarity = slider.value()
                case _param_changed_source.LinearErrorChanged:
                    self._param.linear_error = slider.value()
                case _param_changed_source.DefectMinAreaChanged:
                    self._param.defect_min_area = slider.value()
            self._param_callback(self._param)


    """
    @brief: 向Templates的ComboBox中添加元素的槽函数
    """
    def _add_item_to_templates_combo_box(self, item:str):
        self.TemplatesComboBox.addItem(item)


    def _clear_combo_box(self):
        """
        清除ComboBox中所有元素
        """
        self.TemplatesComboBox.clear()


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
        for view in self._graphic_details_graphic_view_list:
            view.deleteLater()
        self._graphic_details_graphic_view_list = []

        for scene in self._graphic_details_graphic_scene_list:
            scene.clear()
            scene.deleteLater()
        self._graphic_details_graphic_scene_list = []

        for box in self._graphic_details_group_box_list:
            box.deleteLater()
        self._graphic_details_group_box_list = []

        # 绘制新图像
        for title in details:
            img = details[title]

            box_w = self.GraphicDetialList.width() - 30
            box_h = int((self.GraphicDetialList.width() / img.width()) * img.height()) + 30


            # 创建GroupBox
            group_box = QtWidgets.QGroupBox(title, parent=self.GraphicDetialList)
            group_box.setMinimumSize(QtCore.QSize(box_w, box_h))
            group_box.setMaximumSize(QtCore.QSize(box_w, box_h))
            self.GraphicDetialListvLayout.addWidget(group_box)
            self._graphic_details_group_box_list.append(group_box)
        
            # 创建scene
            scene = QGraphicsScene()
            self._graphic_details_graphic_scene_list.append(scene)
            # 创建GraphicView
            graphic_view = QtWidgets.QGraphicsView(group_box)
            graphic_view.setObjectName(title + "GrapgicView")
            graphic_view.setMinimumSize(QtCore.QSize(box_w - 20, box_h - 30))
            graphic_view.setMaximumSize(QtCore.QSize(box_w - 20, box_h - 30))
            graphic_view.move(10, 20)
            graphic_view.setScene(scene)
            graphic_view.wheelEvent = MethodType(self._wheel_event, graphic_view)
            graphic_view.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._graphic_details_graphic_view_list.append(graphic_view)
        
            pixmap = QPixmap.fromImage(img)
            pixmap_item = QGraphicsPixmapItem(pixmap)
            scene.clear()
            scene.addItem(pixmap_item)
            scene.update()


    """
    @brief: 创建MessageBox的槽函数
    """
    def _make_msg_box(self, title:str, text:str):
        QMessageBox.warning(self, title, text)


    """
    @brief: 设置ComboBox选中目标改变的回调函数
    @param:
        - callback: 回调函数, 函数原型应为: `func(id:int, current_name:str)`
    """
    def set_cb_changed_callback(self, target:ComboBoxChangedCallback, callback):
        self._cb_changed_callback_map[target.name] = callback


    """
    @brief: 设置按钮回调函数的接口, 可用于设置 `ButtonCallbackType` 中定义的所有类型回调
    @param:
        - callback: 回调函数, 原型应为 func()
    """
    def set_btn_callback(self, target:ButtonCallbackType, callback):
        self._btn_callback_map[target.name] = callback


    def set_params_changed_callback(self, callback):
        self._param_callback = callback


    def add_template_option(self, option:str):
        """
        @brief: 向模板列表中增加选项
        """
        self._add_item_to_templates_combo_box_signal.emit(option)


    def clear_template_option(self):
        """
        清除向模板列表中的所有选项
        """
        self._clear_combo_box_signal.emit()


    """
    @brief: 设置进度条进度
    @param:
        - target: ProgressBarWidgts中指定的对象枚举
        - value: 进度条值, int, 值域[0, 100]
    """
    def set_progress_bar_value(self, target:ProgressBarWidgts, value:int):
        self._set_progress_bar_value_signal.emit(target, value)


    """
    @brief: 设置图像控件的图像
    """
    def set_graphic_widget(self, cv2_img, target:GraphicWidgets):
        # 获取目标graphic view的size
        target_w = self._graphic_widgets[target.name].width()
        target_h = self._graphic_widgets[target.name].height()
        logging.debug("Widget: %s, size: (%d, %d)"%(target.name, target_w, target_h))
        ## 居中铺满放置
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        y = cv2_img.shape[0]
        x = cv2_img.shape[1]
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


    def make_msg_box(self, title:str, text:str):
        self._make_msg_box_signal.emit(title, text)


