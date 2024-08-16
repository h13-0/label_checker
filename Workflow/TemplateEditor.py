from QtUI.TemplateEditorUI import TemplateEditorUI, TemplateEditorButtonCallbacks, TemplateEditorGraphicViews, EditorUIParams

import os
import time
import threading
import re
import logging
from types import MethodType

import cv2
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QFileDialog, QWidget, QMessageBox

from QtUI.Widgets.MessageBox import MessageBox
from LabelChecker.LabelChecker import LabelChecker
from Utils.Config import Config
from Utils.HSVFilter import HSVFilter
from Template.Template import Template

class TemplateEditor():
    def __init__(self, config:Config, parent:QWidget, template_name:str = "", template_list:list=[]):
        self._ui = TemplateEditorUI(config, template_name, template_list)
        self._config = config
        self._parent = parent
        self._name = template_name
        self._template_list = template_list
        self._window = None

        # 初始化检测器
        self._checker = LabelChecker()
        self._main_thread = None

        # 模板图像及其互斥锁
        self._template_img = None
        self._template_img_id = 0
        self._template_lock = threading.Lock()

        # 输入参数指示变量
        self._input_changed = False
        self._input_changed_lock = threading.Lock()

        # 退出事件
        self._stop_event = threading.Event()
        self._exit_cb = lambda: logging.debug("exit_callback not set.")

        # 加载Template
        self._template = None
        self._shielded_areas = []
        if(len(template_name)):
            ## 导入template
            self._template = self._load_template(os.path.join(config.template_path, template_name))

        # 加载HSVFilter
        self._filter = HSVFilter()
        
        # 设置参数改变回调函数
        self._param = EditorUIParams()
        self._ui.set_params_changed_callback(self._param_changed_cb)


    def show(self):
        self._window = QMainWindow(self._parent)
        self._ui.setupUi(self._window)
        self._window.show()

        # 连接回调函数
        ## 按钮回调函数
        self._ui.set_btn_callback(TemplateEditorButtonCallbacks.OpenTemplatePhotoClicked, self._open_template_photo_cb)

        ## 保存事件回调函数
        self._ui.set_save_template_callback(self._save_callback)

        ## 窗口关闭回调函数
        self._ui.set_window_closed_callback(self._close_event)


    def set_exit_callback(self, callback):
        self._exit_cb = callback


    def _load_template(self, save_path:str) -> Template:
        template = None
        template_img = None
        success = False
        try:
            template = Template.open(save_path)
            template_img = cv2.imread(template.get_img_path())
            self._shielded_areas = template.get_shielded_areas()
            success = True
        except Exception as e:
            logging.error(e)
        
        if(success):
            with self._template_lock:
                self._template_img = template_img
                self._template_img_id += 1
            with self._input_changed_lock:
                    self._input_changed = True


    def _param_changed_cb(self, params:EditorUIParams):
        self._param = params
        with self._input_changed_lock:
            self._input_changed = True


    def _close_event(self):
        """
        @brief: 窗口退出事件回调
        """
        self.exit()


    def _open_file(self):
        """
        @brief: 调用文件选择对话框选择文件
        """
        fname, ftype = QFileDialog.getOpenFileName(self._ui, "Open File", "./", "All Files(*)")
        if(len(fname) == 0):
            logging.warning("No file select.")
        return fname
    
    
    def _open_template_photo_cb(self):
        target_img_file = self._open_file()
        success = False
        if(len(target_img_file)):
            with self._template_lock:
                try:
                    self._template_img = cv2.imread(target_img_file)
                    if(self._template_img is not None):
                        self._template_img_id += 1
                        success = True
                    else:
                        self._ui.make_msg_box("错误", "文件或文件路径错误, 请检查该文件是否为图片或路径是否不包含中文")
                except Exception as e:
                    logging.error(e)
        
            if(success):
                with self._input_changed_lock:
                    self._input_changed = True


    def _process_template(self, 
            template_img, threshold:int, 
            h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int
        ) -> list:
        """
        @brief: 处理模板图像
        @param:
            - template_img: 模板图像
            - threshold: 标签图像黑度阈值
        @return:
            - 返回值为处理后的结果所构成的list
                [
                    template_wraped,
                    template_partten
                ]
        """
        # 1. 先找到标准标签的区域
        min_rect = self._checker.find_label(
            img=template_img, 
            h_min=h_min,
            h_max=h_max, 
            s_min=s_min, 
            s_max=s_max, 
            v_min=v_min, 
            v_max=v_max,
            show_mask = False
        )

        # 2. 将模板标签仿射回标准视角
        temp_wraped = self._checker.wrap_min_aera_rect(template_img, min_rect)

        # 3. 获取标签
        temp_pattern = self._checker.get_pattern(
            wraped_img=temp_wraped,
            threshold=threshold, 
            shielded_areas=None
        )

        return [temp_wraped, temp_pattern]


    def _main(self):
        # 模板原图
        template_img = None
        curr_template_id = 0
        area_inited = False
        while(not self._stop_event.is_set()):
            # 检测是否需要开启下一批计算
            input_changed = False
            with self._input_changed_lock:
                input_changed = self._input_changed
                self._input_changed = False
            
            # 当输入发生变化
            if(input_changed):
                # 输入参数发生变化, 重新执行检测标签
                ## 检测、更新并同步模板图像
                with self._template_lock:
                    if(curr_template_id != self._template_img_id):
                        # 模板图像需要更新
                        template_img = self._template_img.copy()
                
                ## 检查运行条件是否满足
                if(not isinstance(template_img, np.ndarray)):
                    continue

                ## 1. 填装HSVFilter参数
                self._filter.h_high = self._param.h_max
                self._filter.h_low = self._param.h_min
                self._filter.s_high = self._param.s_max
                self._filter.s_low = self._param.s_min
                ## 2. 执行HSVFilter
                filtered = self._filter.filter(template_img)
                ## 3. 将过滤后的图像输出到UI
                self._ui.set_graphic_widget(filtered, TemplateEditorGraphicViews.InputGraphicView)

                template_wraped, template_pattern = self._process_template(
                    template_img=template_img,
                    threshold=170,
                    h_min=self._param.h_min,
                    h_max=self._param.h_max,
                    s_min=self._param.s_min,
                    s_max=self._param.s_max,
                    v_min=0,
                    v_max=255
                )  

                
                self._ui.set_graphic_widget(template_wraped, TemplateEditorGraphicViews.TemplateGraphicView)
            
                if(not area_inited):
                    for area in self._shielded_areas:
                        # 我也不明白为什么不会自动转int, 同时还不报错
                        self._ui.add_shielded_area(area["x1"], area["y1"], area["x2"], area["y2"])


            time.sleep(0.02)
        
        logging.info("Template editor exit.")
        self._exit_cb()


    def _save_callback(self, name:str) -> bool:
        """
        校验模板名并存储
        Return: bool, 保存成功与否
        """
        # 校验模板名
        pattern = r'^[a-zA-Z0-9\-_]+$'
        if(len(name) < 1):
            MessageBox(
                parent=self._ui,
                title="错误",
                content="名称不可为空",
                icon=QMessageBox.Icon.Critical,
                button=QMessageBox.StandardButton.Yes
            ).exec()
            return False
        if(len(name) >= 31):
            MessageBox(
                parent=self._ui,
                title="错误",
                content="名称过长",
                icon=QMessageBox.Icon.Critical,
                button=QMessageBox.StandardButton.Yes
            ).exec()
            return False
        if(not re.match(pattern, name)):
            MessageBox(
                parent=self._ui,
                title="错误",
                content="名称仅可包含字母、数字、减号\"-\"和下划线\"_\"",
                icon=QMessageBox.Icon.Critical,
                button=QMessageBox.StandardButton.Yes
            ).exec()
            return False
        if(
            (name in self._template_list) and
            (name != self._name)
        ):
            responce = MessageBox(
                parent=self._ui,
                title="警告",
                content="已有同名模板, 是否覆盖?",
                icon=QMessageBox.Icon.Question,
                button=QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
            ).exec()
            return responce == QMessageBox.StandardButton.Yes

        # 存储模板配置
        save_path = os.path.join(self._config.template_path, name)
        logging.info("saving " + name + " to " + save_path)
        ## 创建文件夹
        if(not os.path.exists(save_path)):
            try:
                os.makedirs(save_path)
            except Exception as e:
                logging.error(e)
                QMessageBox.critical(self, "错误", str(e))
                return False

        ## 保存图像
        try:
            path = os.path.join(save_path, name + ".jpg")
            cv2.imwrite(path, self._template_img)
        except Exception as e:
            logging.error(e)
            QMessageBox.critical(self, "错误", str(e))
            return False
        ## 创建模板对象
        if(self._template is None):
            self._template = Template(save_path)
        self._template.set_img_type("jpg")
        ## 导出HSV阈值
        self._template.set_hsv_threshold(
            h_min=self._param.h_min,
            h_max=self._param.h_max,
            s_min=self._param.s_min,
            s_max=self._param.s_max,
            v_min=0,#self._param.v_min,
            v_max=255#self._param.v_max,
        )
        ## 导出屏蔽区域和OCR-条码对照区域
        shield_areas = self._ui.get_shield_areas()
        for area in shield_areas:
            x1, y1, x2, y2 = area
            self._template.add_shielded_area(x1, y1, x2, y2)

        ## 保存模板
        try:
            self._template.save()
        except Exception as e:
            logging.error(e)
            QMessageBox.critical(self, "错误", str(e))
            return False
        return True


    def exit(self):
        self._stop_event.set()
        

    def run(self, block:bool = False) -> None:
        """
        @brief: 启动工作流
        @param:
            - block: 是否在当前线程工作, 配合UI使用时应当为 `False`
        """
        if(block):
            self._main()
        else:
            self._main_thread = threading.Thread(target = self._main)
            self._main_thread.start()


    def __del__(self):
        self.exit()
