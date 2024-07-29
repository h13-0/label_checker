from QtUI import UI
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsPixmapItem, QVBoxLayout, QMessageBox, QSpinBox, QLineEdit, QApplication
from PyQt6.QtGui import QPixmap, QImage
import logging
from enum import Enum
import cv2


class ButtonCallbackType(Enum):
    OpenTemplateClicked = 1,
    OpenTargetStreamClicked = 2,
    OpenTargetPhotoClicked = 3,
    LoadParamsClicked = 4,


"""
@brief: 窗口的常驻图像控件(GraphicView)的枚举类
"""
class GraphicWidgets(Enum):
    MainGraphicView = 1,
    TemplateGraphicView = 2


"""
@brief: 参数区运行参数结构体
"""
class WorkingParams():
    def __init__(self, 
        depth_threshold:int = 170, class_similarity:float = 0.1, 
        not_good_similarity:float = 0.1, linear_error:int = 4,
        defect_min_area:int = 100
    ) -> None:
        self.depth_threshold = depth_threshold
        self.class_similarity = class_similarity
        self.not_good_similarity = not_good_similarity
        self.linear_error = linear_error
        self.defect_min_area = defect_min_area



"""
@brief: 私有类, 用于标识参数变化源
"""
class _param_changed_source(Enum):
    DepthThresholdChanged = 1,
    ClassSimilarityChanged = 2,
    NotGoodSimilarityChanged = 3,
    LinearErrorChanged = 4,
    DefectMinAreaChanged = 5


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
        ### 滑动条松开时触发信号
        self.DepthThresholdSlider.sliderReleased.connect(lambda:self._param_changed_cb(_param_changed_source.DepthThresholdChanged))
        self.ClassSimilaritySldider.sliderReleased.connect(lambda:self._param_changed_cb(_param_changed_source.ClassSimilarityChanged))
        self.NotGoodSimilaritySlider.sliderReleased.connect(lambda:self._param_changed_cb(_param_changed_source.NotGoodSimilarityChanged))
        self.LinearErrorSlider.sliderReleased.connect(lambda:self._param_changed_cb(_param_changed_source.LinearErrorChanged))
        self.DefectMinAreaSlider.sliderReleased.connect(lambda:self._param_changed_cb(_param_changed_source.DefectMinAreaChanged))
        ### SpinBox修改时触发信号
        self.DepthThresholdSpinBox.valueChanged.connect(lambda:self._param_changed_cb(_param_changed_source.DepthThresholdChanged))
        self.ClassSimilaritySpinBox.valueChanged.connect(lambda:self._param_changed_cb(_param_changed_source.ClassSimilarityChanged))
        self.NotGoodSimilaritySpinBox.valueChanged.connect(lambda:self._param_changed_cb(_param_changed_source.NotGoodSimilarityChanged))
        self.LinearErrorSpinBox.valueChanged.connect(lambda:self._param_changed_cb(_param_changed_source.LinearErrorChanged))
        self.DefectMinAreaSpinBox.valueChanged.connect(lambda:self._param_changed_cb(_param_changed_source.DefectMinAreaChanged))
        ### 滑动条滑动时, 将值实时更新到SpinBox
        self.DepthThresholdSlider.valueChanged.connect(lambda:self.DepthThresholdSpinBox.setValue(self.DepthThresholdSlider.value()))
        self.ClassSimilaritySldider.valueChanged.connect(lambda:self.ClassSimilaritySpinBox.setValue(self.ClassSimilaritySldider.value()))
        self.NotGoodSimilaritySlider.valueChanged.connect(lambda:self.NotGoodSimilaritySpinBox.setValue(self.NotGoodSimilaritySlider.value()))
        self.LinearErrorSlider.valueChanged.connect(lambda:self.LinearErrorSpinBox.setValue(self.LinearErrorSlider.value()))
        self.DefectMinAreaSlider.valueChanged.connect(lambda:self.DefectMinAreaSpinBox.setValue(self.DefectMinAreaSlider.value()))


        self._update_graphic_signal.connect(self._update_graphic_view, type=Qt.ConnectionType.BlockingQueuedConnection)
        self._set_graphic_detail_signal.connect(self._set_graphic_detail_content, type=Qt.ConnectionType.BlockingQueuedConnection)
        
        self._make_msg_box_signal.connect(self._make_msg_box)
    

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
        changed = False
        # 当Python大于等于3.10时可以尝试使用 match .. case ..
        if(source == _param_changed_source.DepthThresholdChanged):
            # 过滤掉由于Slider改变导致的SpinBox改变引发的信号
            if(
                (QApplication.focusWidget() == self.DepthThresholdSpinBox) or
                (not self.DepthThresholdSlider.isSliderDown())
            ):
                # 如果是SpinBox的更改, 则需要同步到Slider
                if(QApplication.focusWidget() == self.DepthThresholdSpinBox):
                    self.DepthThresholdSlider.setValue(self.DepthThresholdSpinBox.value())
                # 更新数值到params
                self._param.depth_threshold = self.DepthThresholdSlider.value()
                logging.debug("sllider: %d"%(self.DepthThresholdSlider.value()))
                changed = True
        elif(source == _param_changed_source.ClassSimilarityChanged):
            if(
                (QApplication.focusWidget() == self.ClassSimilaritySpinBox) or
                (not self.ClassSimilaritySldider.isSliderDown())
            ):
                # 如果是SpinBox的更改, 则需要同步到Slider
                if(QApplication.focusWidget() == self.ClassSimilaritySpinBox):
                    self.ClassSimilaritySldider.setValue(self.ClassSimilaritySpinBox.value())
                # 更新数值到params
                self._param.class_similarity = self.ClassSimilaritySldider.value()
                logging.debug("sllider: %d"%(self.ClassSimilaritySldider.value()))
                changed = True
        elif(source == _param_changed_source.NotGoodSimilarityChanged):
            if(
                (QApplication.focusWidget() == self.NotGoodSimilaritySpinBox) or
                (not self.NotGoodSimilaritySlider.isSliderDown())
            ):
                # 如果是SpinBox的更改, 则需要同步到Slider
                if(QApplication.focusWidget() == self.NotGoodSimilaritySpinBox):
                    self.NotGoodSimilaritySlider.setValue(self.NotGoodSimilaritySpinBox.value())
                # 更新数值到params
                self._param.not_good_similarity = self.NotGoodSimilaritySlider.value()
                logging.debug("sllider: %d"%(self.NotGoodSimilaritySlider.value())) 
                changed = True
        elif(source == _param_changed_source.LinearErrorChanged):
            if(
                (QApplication.focusWidget() == self.LinearErrorSpinBox) or
                (not self.LinearErrorSlider.isSliderDown())
            ):
                # 如果是SpinBox的更改, 则需要同步到Slider
                if(QApplication.focusWidget() == self.LinearErrorSpinBox):
                    self.LinearErrorSlider.setValue(self.LinearErrorSpinBox.value())
                # 更新数值到params
                self._param.linear_error = self.LinearErrorSlider.value()
                logging.debug("sllider: %d"%(self.LinearErrorSlider.value())) 
                changed = True
        else:
            if(
                (QApplication.focusWidget() == self.DefectMinAreaSpinBox) or
                (not self.DepthThresholdSlider.isSliderDown())
            ):
                # 如果是SpinBox的更改, 则需要同步到Slider
                if(QApplication.focusWidget() == self.DefectMinAreaSpinBox):
                    self.DepthThresholdSlider.setValue(self.DefectMinAreaSpinBox.value())
                # 更新数值到params
                self._param.defect_min_area = self.DepthThresholdSlider.value()
                logging.debug("sllider: %d"%(self.DepthThresholdSlider.value()))
                changed = True

        if(changed):
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


