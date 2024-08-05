import threading
import time
import cv2
import numpy as np
import logging

from PyQt6.QtWidgets import QFileDialog

from Template.Template import Template
from LabelChecker.LabelChecker import LabelChecker
from QtUI.LabelCheckerUI import LabelCheckerUI, ButtonCallbackType, GraphicWidgets, WorkingParams, ComboBoxChangedCallback


"""
@brief: 主工作逻辑
"""
class MainWorkingFlow():
    def __init__(self, ui:LabelCheckerUI, stop_event:threading.Event) -> None:
        self._checker = LabelChecker()
        self._img_list = {}
        self._ui = ui
        self._stop_event = stop_event

        # 运行时参数
        self._params = WorkingParams()
        self._params_lock = threading.Lock()

        # 图像成员变量, 写入dict方便用户debug
        self._img_dict = {}

        ## 模板及其互斥锁
        ### TODO, 弃用dict
        self._img_dict["模板原始图像"] = None
        self._img_dict["模板图像"] = None
        self._img_dict["模板样式"] = None
        ### 模板样式屏蔽区域与OCR-条码核对区域
        ### 模板样式屏蔽区域(包含屏蔽区域和OCR-条码核对区域)
        self._template_shielded_areas = []
        self._template_ocr_bar_code_pairs = []
        ### _template_id 用于检测模板是否变化
        self._template_id = 0
        ### _template_lock 用于保持上述若干变量的互斥访问
        self._template_lock = threading.Lock()

        ## 待检图像及其互斥锁、条件变量。条件变量用于同步视频流等输入
        self._img_dict["待检图像"] = None
        self._img_dict["标签"] = []
        ### _target_id 用于检测目标图像是否变化
        self._target_id = 0
        ### _target_lock 用于保持上述两个图像及id的互斥访问
        self._target_lock = threading.Lock()




        # "输入变化" 事件的条件变量
        self._input_changed = False
        self._input_changed_lock = threading.Lock()


    """
    @brief: 非重要的初始化操作, 可在子线程中执行, 节约主函数等待时间
    """
    def _init(self):
        # 配置回调函数
        self._ui.set_cb_changed_callback(ComboBoxChangedCallback.TemplatesChanged, self._template_changed_cb)
        self._ui.set_btn_callback(ButtonCallbackType.OpenTargetPhotoClicked, self._open_target_photo_cb)
        self._ui.set_params_changed_callback(self._working_param_changed_cb)

        # 扫描模板列表, TODO
        self._ui.add_template_option("创建新模板")
        ## TODO
        self._template_dict = {
            "600PPI_With_SN": Template.open("../templates/600PPI_With_SN")
        }
        for template in self._template_dict:
            self._ui.add_template_option(template)
        

    """
    @brief: 调用文件选择对话框选择文件
    """
    def _open_file(self):
        fname, ftype = QFileDialog.getOpenFileName(self._ui, "Open File", "./", "All Files(*)")
        if(len(fname) == 0):
            logging.warning("No file select.")
        return fname


    """
    @brief: 当目标模板更改时触发该回调函数
    """
    def _template_changed_cb(self, id:int, curr_template:str):
        logging.info("template target changed to: %s, id: %d", curr_template, id)
        success = False
        if(id > 0):
            template = self._template_dict[curr_template]
            logging.info("template config: " + str(template))
            with self._template_lock:
                try:
                    # 读取图像
                    self._img_dict["模板原始图像"] = cv2.imread(template.get_img_path())
                    # 导入屏蔽区域
                    for area in template.get_shielded_areas():
                        self._template_shielded_areas.append((
                            area["x1"], area["y1"], area["x2"], area["y2"]
                        ))
                    # 导入OCR-条码核对区
                    ## TODO
                    
                    # 校验结果
                    if(self._img_dict["模板原始图像"] is not None):
                        self._template_id += 1
                        success = True
                    else:
                        self._ui.make_msg_box("错误", "文件或文件路径错误, 请检查该文件是否为图片或路径是否不包含中文")
                except Exception as e:
                    logging.error(e)

            if(success):
                with self._input_changed_lock:
                    self._input_changed = True


    """
    @brief: "打开待检图像"的回调函数
    """
    def _open_target_photo_cb(self):
        target_img_file = self._open_file()
        success = False
        if(len(target_img_file)):
            with self._target_lock:
                try:
                    self._img_dict["待检图像"] = cv2.imread(target_img_file)
                    if(self._img_dict["待检图像"] is not None):
                        self._target_id += 1
                        success = True
                    else:
                        self._ui.make_msg_box("错误", "文件或文件路径错误, 请检查该文件是否为图片或路径是否不包含中文")
                except Exception as e:
                    logging.error(e)
        
            if(success):
                with self._input_changed_lock:
                    self._input_changed = True


    """
    @brief: "参数调整区"的回调函数
    """
    def _working_param_changed_cb(self, params:WorkingParams):
        with self._params_lock:
            self._params = params
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


    """
    @brief: 从待测图像中寻找标签
    @param:
        - img: 待测图像
        - template_w: 模板宽度
        - template_h: 模板高度
        - wh_tol_ratio: 容许的长宽误差比例
    @return:
        - 返回值为由minAreaRect组成的list
    """
    def _find_labels(self, 
        img, template_w:int, template_h:int,
        h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int,
        wh_tol_ratio:float = 0.1
    ) -> list:
        rects = self._checker.find_labels(
            img=img,
            template_w=template_w,
            template_h=template_h,
            h_min=h_min,
            h_max=h_max,
            s_min=s_min,
            s_max=s_max,
            v_min=v_min,
            v_max=v_max,
            wh_tol_ratio=wh_tol_ratio,
            show_mask=False
        )
        
        logging.info("total %d labels find", len(rects))
        return rects


    """
    @brief: 将target_img图像中指定的target_rect所在的标签与template_pattern进行匹配
    @param:
        - template_pattern: [只读参数], 模板样式
        - target_img: [只读参数], 待测图像, 可以包含多个标签
        - target_rect: [只读参数], 目标标签所在minAreaRect
        - threshold: [只读参数], 标签图像黑度阈值, 同方法 `_process_template` 中的同名参数
        - initial_minimum_simi: [只读参数], 初始最低相似度, 定义为: sim = (匹配样式像素数) / (标准样式像素数)
        - 
        - thickness_tol: [只读参数], 容许的粗细误差
    @return:
        - [ 二值图样式, 二值图对应的彩色图 ]
    @note: 
        本函数未来会做为并行运算的方法使用
    """
    def _match_label(self, 
        template_pattern, target_img, target_rect, threshold:int, shielded_areas:list,
        initial_minimum_simi:float = 0.01, thickness_tol:int = 3
    ):
        # 1. 将模板标签仿射回标准视角
        target_wraped = self._checker.wrap_min_aera_rect(target_img, target_rect)

        # 2. 获取样式
        pattern = self._checker.get_pattern(target_wraped, threshold)

        # 3. 监测相似度是否超标
        loss = self._checker.try_match(
            pattern, template_pattern, 
            x=0, 
            y=0, 
            angle=0,
            shielded_areas=shielded_areas,
            show_diff=False
        )
        
        # 4. TODO: 判定初始相似度是否超标
        logging.info("初始误差: " + str(loss))

        # 5. 微调, TODO: 参数可调
        x, y, angle = self._checker.fine_tune(
            test=pattern, std=template_pattern,
            max_abs_x=10, max_abs_y=10, max_abs_a=1,
            max_iterations=40, 
            shielded_areas=shielded_areas,
            angle_step=0.0015, view_size=2,
            show_process=True
        )

        # 6. 获取微调后的样式
        target_pattern = self._checker.linear_trans_to(
            img=pattern, x=x, y=y, angle=angle, output_size=[template_pattern.shape[1], template_pattern.shape[0]], border_color=0
        )

        # 7. 获得误差图像
        target_remain = self._checker.cut_with_tol(template_pattern, target_pattern, thickness_tol, shielded_areas)
        template_remain = self._checker.cut_with_tol(target_pattern, template_pattern, thickness_tol, shielded_areas)
        diff = cv2.bitwise_or(target_remain, template_remain)

        # 8. 计算线性变换后原图
        target_trans = self._checker.linear_trans_to(
            img=target_wraped, x=x, y=y, angle=angle, output_size=[template_pattern.shape[1], template_pattern.shape[0]], border_color=[255, 255, 255]
        )
        return [ diff, target_trans, pattern ]



    def _main(self):
        self._init()

        # 模板原图
        template_img = None
        ## 仿射后的标签
        template_wraped = None
        template_pattern = None
        ## 样式屏蔽区域及OCR-条码核对区
        template_shielded_areas = []
        template_ocr_bar_code_pairs = []
        template_pattern_size = 0
        template_w = 0
        template_h = 0

        # 目标原图
        target_img = None

        # 最终错误方框粗细
        box_thickness = 3

        curr_template_id = 0
        curr_target_id = 0
        # 检测退出标志
        while(not self._stop_event.is_set()):
            # 检测是否需要开启下一批计算
            input_changed = False
            with self._input_changed_lock:
                input_changed = self._input_changed
                self._input_changed = False
                
            
            # 当输入变化时, 重新运算
            if(input_changed):
                params = None
                # 拷贝一份参数
                with self._params_lock:
                    params = self._params

                # 输入参数发生变化, 重新执行检测标签
                ## 检测、更新并同步模板图像
                with self._template_lock:
                    if(curr_template_id != self._template_id):
                        # 模板需要更新
                        template_img = self._img_dict["模板原始图像"].copy()
                        self._img_dict["模板图像"], self._img_dict["模板样式"] = self._process_template(
                            template_img=template_img,
                            threshold=params.depth_threshold,
                            h_min=params.h_min,
                            h_max=params.h_max,
                            s_min=params.s_min,
                            s_max=params.s_max,
                            v_min=0,
                            v_max=255
                        )
                        template_wraped = self._img_dict["模板图像"]

                        template_shielded_areas = self._template_shielded_areas.copy()
                        template_ocr_bar_code_pairs = self._template_ocr_bar_code_pairs.copy()

                        template_pattern = self._checker.get_pattern(template_wraped, params.depth_threshold)
                        template_pattern_size = cv2.countNonZero(template_pattern)
                        logging.debug("@main:template_pattern_size: %d"%(template_pattern_size))



                        template_w = template_wraped.shape[1]
                        template_h = template_wraped.shape[0]
                        # 在UI中更新图像
                        self._ui.set_graphic_widget(self._img_dict["模板图像"], GraphicWidgets.TemplateGraphicView)
                        curr_template_id = self._template_id

                ## 检测、更新并同步待测图像
                with self._target_lock:
                    if(curr_target_id != self._target_id):
                        # 目标需要更新
                        target_img = self._img_dict["待检图像"].copy()

                ## 检查运行条件是否满足
                if(not isinstance(template_img, np.ndarray)):
                    continue

                if(not isinstance(target_img, np.ndarray)):
                    continue

                ## 开始对待检图像进行图像处理 TODO
                rects = self._find_labels(
                    target_img,
                    template_w=template_w,
                    template_h=template_h,
                    h_min=params.h_min,
                    h_max=params.h_max,
                    s_min=params.s_min,
                    s_max=params.s_max,
                    v_min=0,
                    v_max=255,
                    wh_tol_ratio=0.1
                )

                # 匹配并绘制标签
                target_img_with_mark = target_img.copy()
                id = 0
                ## 存储结果
                target_result = {}
                for r in rects:
                    # 匹配标签并获得缺陷图
                    diff, target_trans, pattern = self._match_label( 
                        template_pattern=template_pattern,
                        target_img=target_img,
                        target_rect=r,
                        threshold=params.depth_threshold,
                        shielded_areas=template_shielded_areas,
                        thickness_tol=params.linear_error
                    )

                    # 计算误差
                    loss = cv2.countNonZero(diff)
                    ## 判断阈值
                    similarity = (template_pattern_size - loss)/float(template_pattern_size)
                    logging.info("label: %d, final_loss: %d, similarity: %f"%(id, loss, similarity))
                    ## 绘制方框
                    if(similarity * 100 < params.class_similarity):
                        # 不同类标签
                        target_img_with_mark = cv2.drawContours(target_img_with_mark, [np.int_(cv2.boxPoints(r))], 0, (0, 0, 255), 2)
                    elif(similarity * 100 < params.not_good_similarity):
                        # 同类较差标签
                        target_img_with_mark = cv2.drawContours(target_img_with_mark, [np.int_(cv2.boxPoints(r))], 0, (0, 0, 0), 2)
                    else:
                        # 同类较优标签
                        target_img_with_mark = cv2.drawContours(target_img_with_mark, [np.int_(cv2.boxPoints(r))], 0, (0, 255, 0), 2)
                    
                    ## 绘制标签
                    target_img_with_mark = cv2.putText(
                        img=target_img_with_mark, 
                        text="id: " + str(id), 
                        org=(int(r[0][0] - template_w / 2), int(r[0][1] - template_h / 2)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=0.75, 
                        color=(0, 0, 255), 
                        thickness=2
                    )

                    # 绘制误差点并输出图像
                    if(similarity * 100 > params.class_similarity):
                        # 同类标签中绘制误差点
                        contours, hierarchy = cv2.findContours(diff, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                        for c in contours:
                            x, y, w, h = cv2.boundingRect(c)
                            size = w * h
                            if(size > params.defect_min_area):
                                cv2.rectangle(
                                    target_trans, 
                                    (x - box_thickness, y - box_thickness), 
                                    (x + w + box_thickness, y + h + box_thickness), 
                                    (0, 0, 255), 
                                    box_thickness
                                )
                            logging.debug("label: %d, defect size: %d"%(id, w * h))
                        # show
                        target_result["id: " + str(id)] = target_trans
                        diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
                        target_result["id: " + str(id) + " diff"] = diff_bgr
                        pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
                        target_result["id: " + str(id) + " pattern"] = pattern_bgr

                    id += 1
                
                # 在操作ui之前确保程序没有被退出
                if(not self._stop_event.is_set()):
                    self._ui.set_graphic_detail_list(target_result)
                    self._ui.set_graphic_widget(target_img_with_mark, GraphicWidgets.MainGraphicView)
                else:
                    logging.info("main workflow exit.")
            
            # Sleep 0.02s
            cv2.waitKey(2)
            #time.sleep(0.02)
        logging.info("main workflow exit.")


    def Run(self, block:bool = False) -> None:
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

