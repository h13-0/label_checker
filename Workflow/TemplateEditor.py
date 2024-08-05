from QtUI.TemplateEditorUI import TemplateEditorUI, TemplateEditorButtonCallbacks, TemplateEditorGraphicViews

import time
import threading
import logging
from types import MethodType

import cv2
import numpy as np

from PyQt6.QtWidgets import QMainWindow, QFileDialog

from LabelChecker.LabelChecker import LabelChecker

class TemplateEditor():
    def __init__(self):
        self._ui = TemplateEditorUI()
        self._window = None

        self._checker = LabelChecker()
        self._main_thread = None

        # 模板图像及其互斥锁
        self._template_img = None
        self._template_img_id = 0
        self._template_lock = threading.Lock()

        # 输入参数指示变量
        self._input_changed = False
        self._input_changed_lock = threading.Lock()

        # 连接回调函数
        self._ui.set_btn_callback(TemplateEditorButtonCallbacks.OpenTemplatePhotoClicked, self._open_template_photo_cb)

        # 退出事件
        self._stop_event = threading.Event()


    def show(self):
        if(self._window is None):
            self._window = QMainWindow()
            self._ui.setupUi(self._window)
            self._window.closeEvent = MethodType(self._close_event, self._window)
            self._window.show()


    """
    @brief: 窗口退出事件回调
    """
    def _close_event(self, widget, event):
        self.exit()
        event.accept()


    """
    @brief: 调用文件选择对话框选择文件
    """
    def _open_file(self):
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
    def _process_template(self, 
            template_img, threshold:int, 
            h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int
        ) -> list:
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

        # 3. 将图像反色
        temp_wraped_reversed = cv2.bitwise_not(temp_wraped)

        # 4. 将标签图像转为二值图
        temp_pattern = cv2.threshold(cv2.cvtColor(temp_wraped_reversed, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

        return [temp_wraped, temp_pattern]


    def _main(self):
        # 模板原图
        template_img = None
        curr_template_id = 0
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

                template_wraped, template_pattern = self._process_template(
                    template_img=template_img,
                    threshold=170,
                    h_min=0,
                    h_max=170,
                    s_min=13,
                    s_max=255,
                    v_min=0,
                    v_max=255
                )  

                self._ui.set_graphic_widget(template_wraped, TemplateEditorGraphicViews.TemplateGraphicView)
                
            time.sleep(0.02)


    def exit(self):
        self._stop_event.set()
        logging.info("Template editor exit.")


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
