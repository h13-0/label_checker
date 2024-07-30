from QtUI import UI
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QMessageBox, QSpinBox, QLineEdit, QApplication
from PyQt6.QtGui import QPixmap, QImage
import logging
from enum import Enum
import cv2
from functools import partial


class ButtonCallbackType(Enum):
    OpenTemplateClicked = 1
    OpenTargetStreamClicked = 2
    OpenTargetPhotoClicked = 3
    LoadParamsClicked = 4


"""
@brief: 窗口的常驻图像控件(GraphicView)的枚举类
"""
class GraphicWidgets(Enum):
    MainGraphicView = 1
    TemplateGraphicView = 2


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
        depth_threshold:int = 170, class_similarity:int = 90, 
        not_good_similarity:int = 95, linear_error:int = 5,
        defect_min_area:int = 5
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


class Ui_Main(UI.Ui_LabelChecker, QWidget):
    # 定义Qt信号
    _update_graphic_signal = pyqtSignal(QImage, GraphicWidgets)
    _set_graphic_detail_signal = pyqtSignal(dict)
    _make_msg_box_signal = pyqtSignal(str, str)

    def __init__(self):
        UI.Ui_LabelChecker.__init__(self)
        QWidget.__init__(self)

        self._btn_callback_map = {}
        self._param_callback = lambda: logging.warning("params_changed_callback not set.")

        # 默认参数
        self._param = WorkingParams()

    
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
        ## 按钮
        self.OpenTemplateButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.OpenTemplateClicked))
        self.OpenTargetStreamButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.OpenTargetStreamClicked))
        self.OpenTargetPhotoButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.OpenTargetPhotoClicked))
        self.LoadParamsButton.clicked.connect(lambda:self._btn_callbacks(ButtonCallbackType.LoadParamsClicked))

        ## 参数区回调函数
        ### 创建参数枚举和参数控件的映射
        self._param_widgets = {
            _param_changed_source.HMinChagned: [self.HMinSlider, self.HMinSpinBox],
            _param_changed_source.HMaxChanged: [self.HMaxSlider, self.HMaxSpinBox],
            _param_changed_source.SMinChanged: [self.SMinSlider, self.SMinSpinBox],
            _param_changed_source.SMaxChanged: [self.SMaxSlider, self.SMaxSpinBox],
            _param_changed_source.DepthThresholdChanged: [self.DepthThresholdSlider, self.DepthThresholdSpinBox],
            _param_changed_source.ClassSimilarityChanged: [self.ClassSimilaritySldider, self.ClassSimilaritySpinBox],
            _param_changed_source.NotGoodSimilarityChanged: [self.NotGoodSimilaritySlider, self.NotGoodSimilaritySpinBox],
            _param_changed_source.LinearErrorChanged: [self.LinearErrorSlider, self.LinearErrorSpinBox],
            _param_changed_source.DefectMinAreaChanged: [self.DefectMinAreaSlider, self.DefectMinAreaSpinBox],
        }
        self._connect_param_widgets_signal(self._param_widgets)

        ### 滑动条滑动时, 将值实时更新到SpinBox, TODO: 并入self._connect_param_widgets_signal
        self.HMinSlider.valueChanged.connect(lambda:self.HMinSpinBox.setValue(self.HMinSlider.value()))
        self.HMaxSlider.valueChanged.connect(lambda:self.HMaxSpinBox.setValue(self.HMaxSlider.value()))
        self.SMinSlider.valueChanged.connect(lambda:self.SMinSpinBox.setValue(self.SMinSlider.value()))
        self.SMaxSlider.valueChanged.connect(lambda:self.SMinSpinBox.setValue(self.SMaxSlider.value()))
        self.DepthThresholdSlider.valueChanged.connect(lambda:self.DepthThresholdSpinBox.setValue(self.DepthThresholdSlider.value()))
        self.ClassSimilaritySldider.valueChanged.connect(lambda:self.ClassSimilaritySpinBox.setValue(self.ClassSimilaritySldider.value()))
        self.NotGoodSimilaritySlider.valueChanged.connect(lambda:self.NotGoodSimilaritySpinBox.setValue(self.NotGoodSimilaritySlider.value()))
        self.LinearErrorSlider.valueChanged.connect(lambda:self.LinearErrorSpinBox.setValue(self.LinearErrorSlider.value()))
        self.DefectMinAreaSlider.valueChanged.connect(lambda:self.DefectMinAreaSpinBox.setValue(self.DefectMinAreaSlider.value()))
        

        #self.DepthThresholdSlider.valueChanged.connect(lambda:self.DepthThresholdSpinBox.setValue(self.DepthThresholdSlider.value()))

        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._set_graphic_detail_signal.connect(self._set_graphic_detail_content, type=Qt.ConnectionType.BlockingQueuedConnection)
        
        self._make_msg_box_signal.connect(self._make_msg_box)
    

    def _call_test(self, spinbox):
        print("call, spin: " + str(spinbox.value()))


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
    @brief: 按钮的槽函数, 回调事件转发器
    """
    def _btn_callbacks(self, target:ButtonCallbackType):
        if(not target.name in self._btn_callback_map):
            logging.warning("Callback: " + target.name + " not set.")
        else:
            self._btn_callback_map[target.name]()


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


    """
    @brief: 创建MessageBox的槽函数
    """
    def _make_msg_box(self, title:str, text:str):
        QMessageBox.warning(self, title, text)



    """
    @brief: 设置按钮回调函数的接口, 可用于设置 `ButtonCallbackType` 中定义的所有类型回调
    """
    def set_btn_callback(self, target:ButtonCallbackType, callback):
        self._btn_callback_map[target.name] = callback


    def set_params_changed_callback(self, callback):
        self._param_callback = callback


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


    def make_msg_box(self, title:str, text:str):
        self._make_msg_box_signal.emit(title, text)


