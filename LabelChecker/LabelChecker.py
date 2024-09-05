import numpy as np
import cv2
import math
import logging


class LabelChecker():
    def __init__(self) -> None:
        pass


    def find_label(self, 
        img:np.ndarray, h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int,
        show_mask:bool = False
    ):
        '''
        @brief: 从图像中寻找标签
        @note: 输入图像应当仅包含一个标签
        @return: minAreaRect
        '''
        # 使用HSV提取标签所在色块
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        if(show_mask):
            cv2.imshow("@find_label:mask", mask)

        # 寻找最大区域
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        max_size = -1
        max_area_rect = None
        for c in contours:
            minRect  = cv2.minAreaRect(c)
            #minRect = self.fix_min_aera_rect(minRect)
            size = minRect[1][0] * minRect[1][1]
            if(size > max_size):
                max_size = size
                max_area_rect = minRect

        return max_area_rect


    def find_labels(self, 
        img:np.ndarray, template_w:int, template_h:int,
        h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int,
        wh_tol_ratio:float = 0.1, show_mask:bool = False
    ):
        '''
        @brief: 从图像中寻找所有标签
        @param:
            - img: 输入图像
            - template_w: 模板宽度
            - template_h: 模板高度
            - wh_tol_ratio: 容许的长宽误差比例
            - ed_kernel_size: 腐蚀膨胀核大小
            - ed_iterations: 腐蚀膨胀代数
        @return: minAreaRect
        '''
        # 使用HSV提取标签所在色块
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        if(show_mask):
            cv2.imshow("@find_label:mask", mask)

        # 寻找标签所在区域
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        rects = []
        for c in contours:
            minRect = cv2.minAreaRect(c)
            #minRect = self.fix_min_aera_rect(minRect)
            w = max(minRect[1][0], minRect[1][1])
            h = min(minRect[1][0], minRect[1][1])
            #h = minRect[1][1]
            if(
                abs((w / float(template_w)) - 1) < wh_tol_ratio and
                abs((h / float(template_h)) - 1) < wh_tol_ratio
            ):
                rects.append(minRect)

        return rects


    def fix_min_aera_rect(self, min_area_rect:tuple) -> tuple:
        '''
        @brief: 该函数主要用于修复角度错误的问题
        @return:
            ((x, y), (w, h), angle)
        @note: angle定义与opencv中一致, 为从x方向逆时针旋转所重合到的第一条边的角度
        '''
        tl, tr, bl, br = self.get_box_point(min_area_rect)

        x = (tl[0] + tr[0] + bl[0] + br[0]) / 4.0
        y = (tl[1] + tr[1] + bl[1] + br[1]) / 4.0
        w = math.sqrt(math.pow(tl[0] - tr[0], 2) + math.pow(tl[1] - tr[1], 2))
        h = math.sqrt(math.pow(tl[0] - bl[0], 2) + math.pow(tl[1] - bl[1], 2))
        angle = 0
        try:
            angle = math.acos((tr[0] - tl[0]) / w) / math.pi * 180.0
        except Exception as e:
            angle = math.nan
        return ((x, y), (w, h), angle)


    def get_box_point(self, min_area_rect:tuple) -> list:
        """
        @brief: 将min_area_rect转换为四个顶点, 横方向为长边并且输出顺序为: [左上 右上 左下 右下],
        @return:
            - 横方向为长边并且输出list为[左上 右上 左下 右下]
        """
        # cv2.boxPoints得到的四个点一定是按照顺时针排序的
        points = cv2.boxPoints(min_area_rect)

        # 先计算前两个边的边长, 从而确定长边
        side1 = math.sqrt(
            math.pow(points[0][0] - points[1][0], 2) + 
            math.pow(points[0][1] - points[1][1], 2)
        )
        side2 = math.sqrt(
            math.pow(points[1][0] - points[2][0], 2) + 
            math.pow(points[1][1] - points[2][1], 2)
        )

        result = []
        if(side1 > side2):
            # 第一边大于第二边, 则第(0-1)、(2-3)边为长边
            ## 判断0、3点高度
            if(points[0][1] < points[3][1]):
                # 此时0点为左上点
                result = [points[0], points[1], points[3], points[2]]
            else:
                # 此时2点为左上点
                result = [points[2], points[3], points[1], points[0]]
        else:
            # 第二边大于第一边, 则第(1-2)、(3-4)边为长边
            ## 判断0、1点高度
            if(points[0][1] < points[1][1]):
                # 此时3点为左上点
                result = [points[3], points[0], points[2], points[1]]
            else:
                # 此时1点为左上点
                result = [points[1], points[2], points[0], points[3]]
        return result


    def wrap_min_aera_rect(self, src:np.ndarray, min_area_rect):
        '''
        @brief: 将 minAreaRect 所在区域仿射并裁切到长方形
        @note: 该函数会自动将图像旋转到正确的方向
        '''
        # 提取w、h
        w = min_area_rect[1][0]
        h = min_area_rect[1][1]
        if(w < h):
            w = min_area_rect[1][1]
            h = min_area_rect[1][0]

        # 计算仿射前四个顶点位置
        points = self.get_box_point(min_area_rect)
        src_array = np.array(
            [
                # 左上、右上
                points[0], points[1],
                # 左下、右下
                points[2], points[3],
            ], 
            dtype='float32'
        )
        dst_array = np.array(
            [
                [0, 0], [w - 1, 0],
                [0, h - 1], [w - 1, h - 1]
            ], 
            dtype='float32'
        )
        matrix = cv2.getPerspectiveTransform(src_array, dst_array)
        return cv2.warpPerspective(src, matrix, (int(w), int(h)))


    def get_pattern(self, wraped_img:np.ndarray, threshold:int, shielded_areas:list=None):
        """
        @brief: 将经过仿射变换的标签图像转化为样式二值图
        @param:
            - wraped_img: 仿射后的图像
            - threshold: 黑度阈值(255: 最黑, 0: 最亮)
            - shielded_areas: 需要屏蔽的区域, 在这些区域内的样式二值图会被覆盖为黑色(即屏蔽对应区域)。为list类型, 数据结构定义如下: 
                [
                    (x1, y1, x2, y2),
                    (...)
                ]
        """
        # 1. 将图像反色
        target_wraped_reversed = cv2.bitwise_not(wraped_img)

        # 2. 将标签图像转为二值图
        _, target_pattern = cv2.threshold(cv2.cvtColor(target_wraped_reversed, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

        # 3. 对选定区域进行屏蔽
        if(isinstance(shielded_areas, list)):
            for area in shielded_areas:
                # 检查输入维度是否合法
                if(len(area) != 4):
                    logging.error("Shielded'area setting error, requires [(x1, x2, y1, y2), ...], But get [..., " + str(area) + ", ...]")
                    continue
                # 检查坐标点是否合法
                if(area[2] < area[0] or area[3] < area[1]):
                    logging.error("invalid shielded_area: " + str(area))
                    continue
                # 检查坐标点是否越界
                if(
                    area[0] > wraped_img.shape[1] or area[2] > wraped_img.shape[1] or
                    area[1] > wraped_img.shape[0] or area[3] > wraped_img.shape[0]
                ):
                    logging.error("rect point out of bounds: " + str(area))
                    continue
                # 执行屏蔽
                target_pattern = cv2.rectangle(target_pattern, (area[0], area[1]), (area[2], area[3]), 0, thickness=cv2.FILLED)

        return target_pattern


    def linear_trans_to(self, img:np.ndarray, x:int, y:int, angle:float, output_size:list = [-1, -1], border_color = [0, 0, 0]):
        '''
        @brief: 将图像进行线性变换, 边缘用黑色填充
        @param:
            - img: 待变换图像
            - x: x轴offset
            - y: y轴offset
            - angle: 旋转角
            - output_size: 输出图像大小, 默认为原大小, 保持中心对齐。大则裁切小则填充。
                output_size[0]: w
                output_size[1]: h
                若 output_size = [-1, -1] 则不改变输出图像大小
            - border_color: 边缘填充颜色, 注意图像维度, 三色图应为 [B, G, R], 单色图为 value
        @return:
            线性变换并保持中心对齐后, 经过裁切或填充的图像。
        '''
        if(output_size == [-1, -1]):
            output_size[0] = img.shape[1]
            output_size[1] = img.shape[0]
        # 备份处理前待测图像参数
        ori_w = img.shape[1]
        ori_h = img.shape[0]
        out_cen_x = output_size[0] / 2.
        out_cen_y = output_size[1] / 2.
        ## 对角长度(向上取整)
        ori_hypo = math.ceil(math.sqrt(pow(ori_w, 2) + pow(ori_h, 2)))

        # 为待测图像扩展黑边
        border_w = math.ceil((ori_hypo - ori_w) / 2) + abs(x)
        border_h = math.ceil((ori_hypo - ori_h) / 2) + abs(y)
        test_ext = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=border_color)
        ext_w = test_ext.shape[1]
        ext_h = test_ext.shape[0]

        # 对test图像进行旋转
        matrix = cv2.getRotationMatrix2D(center=(ext_w / 2, ext_h / 2), angle=angle, scale=1)
        test_rot = cv2.warpAffine(test_ext, matrix, (ext_w, ext_h))

        # 从中心裁切到与输出大小同大, 并在裁切时位移
        rot_w = test_rot.shape[1]
        rot_h = test_rot.shape[0]
        rot_cen_x = rot_w / 2.
        rot_cen_y = rot_h / 2.
        x_min = int(rot_cen_x - out_cen_x + x)
        x_max = int(rot_cen_x + out_cen_x + x)
        y_min = int(rot_cen_y - out_cen_y + y)
        y_max = int(rot_cen_y + out_cen_y + y)
        trans = test_rot[ y_min : y_max, x_min : x_max ]
        if(trans.shape[0] != output_size[1] or trans.shape[1] != output_size[0]):
            trans = cv2.resize(trans, (output_size[0], output_size[1]))
        return trans


    def try_match(self, img1:np.ndarray, img2:np.ndarray, x:int, y:int, angle, shielded_areas:list=None, show_diff = False) -> int:
        '''
        @brief: 将待测图像以指定线性变换匹配到标准图像上, 用于二值图匹配
        @param:
            - img1: 待变换的图像
            - img2: 欲匹配的图像
        @return: 未能匹配的像素数量
        '''
        # 线性变换
        trans = self.linear_trans_to(img1, x, y, angle, [img2.shape[1], img2.shape[0]], 0)
        # 对选定区域进行屏蔽
        if(isinstance(shielded_areas, list)):
            for area in shielded_areas:
                # 执行屏蔽
                trans = cv2.rectangle(trans, (area[0], area[1]), (area[2], area[3]), 0, thickness=cv2.FILLED)
        # 计算像素点误差
        diff = cv2.absdiff(trans, img2)
        if(show_diff):
            cv2.imshow("try_match:trans", trans)
            cv2.imshow("try_match:diff", diff)
            cv2.waitKey(1)
        loss = cv2.countNonZero(diff)
        return loss



    def fine_tune(self, test:np.ndarray, std:np.ndarray, max_abs_x:int, max_abs_y:int, max_abs_a:float, max_iterations, 
        shielded_areas:list=None, angle_accu = 0.1, view_size:int = 2, show_process:bool = False
    ):
        '''
        @brief: 将待测图像(test)通过线性变换微调到模板(std)上。
        @param:
            - test
            - std
            - max_abs_x: x轴最大微调像素数(绝对值)
            - max_abs_y: y轴最大微调像素数(绝对值)
            - max_abs_a: 最大旋转角角度数(绝对值)
            - max_iterations: 最大微调迭代数
            - angle_accu: 旋转角调整精度, 当angle_accu<0时为跳过旋转角匹配
            - view_size: 视野边距: 
                例如边距为为1时, 对应视野矩阵为3x3, 视野行向量长度为3
                例如边距为为2时, 对应视野矩阵为5x5, 视野行向量长度为5
        @return:
            best param in [x, y, angle]
        @note: 本函数只能处理微小误差(半行以内), 即初始时应当已几乎匹配
        '''
        iterations = 0
        # 初始值与现值
        curr_x = 0
        curr_y = 0
        curr_a = 0
        ## [x, y, angle]
        last_params = [0, 0, 0.]
        last_loss = self.try_match(
            test, std, 
            x=0, 
            y=0, 
            angle=0, 
            shielded_areas=shielded_areas
        )
        best_params = last_params
        best_loss = last_loss
        logging.info("iteration: %d, loss: %f" % (iterations, last_loss))

        # 矩阵大小
        matrix_size = view_size * 2 + 1
        ## xy损失矩阵, 未计算点位为-1
        xy_loss = np.full((matrix_size, matrix_size), -1)
        # 旋转角度损失矩阵, 未计算点为-1
        angle_loss = np.full(matrix_size, -1)
        curr_angle_step = max_abs_a / float(view_size)

        # 定义结束变量
        ## 基本逻辑：
        ### 1. 匹配xy offset至最优, 并记录在匹配过程中xy offset是否改变
        ### 2. 匹配angle offset至最优, 并记录在匹配过程中angle offset是否改变
        ### 3. 若xy offset、angle offset均未改变, 则抵达全局最优, 结束运行。否则回到步骤1。
        ## 变量含义
        ### complete用于指示xy和angle单次匹配是否结束
        xy_complete = False
        angle_complete = False
        ### changed用于指示offset是否改变
        xy_changed = False
        angle_changed = False
        while(iterations < max_iterations):
            iterations += 1
            
            if(xy_complete == False):
                # 更新loss矩阵, 并寻找loss最低值
                ## min_dxy存放的是相对于上一轮局部最优点坐标的相对坐标, 例如:
                ##  - 上一轮局部最优坐标点绝对值为(1, 1), 本轮局部最优坐标点为(-1, 2), 则min_dxy=[-2, 1]
                min_dxy = [0, 0]
                min_loss = -1

                for i in range(matrix_size):
                    for j in range(matrix_size):
                        if(xy_loss[i][j] < 0):
                            xy_loss[i][j] = self.try_match(
                                test, std, 
                                x=curr_x + j - view_size, 
                                y=curr_y + i - view_size, 
                                angle=curr_a, 
                                shielded_areas=shielded_areas
                            )
                        if((min_loss < 0) or (xy_loss[i][j] < min_loss)):
                            min_loss = xy_loss[i][j]
                            min_dxy = [ j - view_size, i - view_size ]
                
                logging.debug("xy_loss matrix: %s"%(str(xy_loss)))
                logging.debug("minxy: %s"%(str(min_dxy)))
                logging.debug("minloss: %d"%(min_loss))

                # 当中心也为最小值时则不移动, 同时停止匹配
                if(xy_loss[view_size][view_size] == min_loss):
                    min_dxy = [ 0, 0 ]
                    xy_complete = True

                # 向最小loss方向移动
                next_x = curr_x + min_dxy[0]
                next_y = curr_y + min_dxy[1]
                # 限幅
                if(next_x < - max_abs_x):
                    next_x = - max_abs_x
                elif(next_x > max_abs_x):
                    next_x = max_abs_x
                
                if(next_y < - max_abs_y):
                    next_y = - max_abs_y
                elif(next_y > max_abs_y):
                    next_y = max_abs_y
                
                delta_x = next_x - curr_x
                delta_y = next_y - curr_y
                
                # 检测xy offset是否发生变化
                if(delta_x == 0 and delta_y == 0):
                    xy_changed = False
                else:
                    xy_changed = True

                # 将x、y指向loss更低的位置, 并更新loss矩阵(用-1填充矩阵)
                if(delta_x < 0):
                    # 中心向左移动abs(delta_x)个坐标点, 矩阵向右移动abs(delta_x)个坐标点
                    ## 仍有效的矩阵remain_xy=xy_loss[:, : matrix_size + delta_x]
                    new_loss = np.full((matrix_size, matrix_size), -1)
                    new_loss[: , abs(delta_x):] = xy_loss[:, : matrix_size + delta_x]
                    xy_loss = new_loss
                elif(delta_x > 0):
                    # 中心向右移动abs(delta_x)个坐标点, 矩阵向左移动abs(delta_x)个坐标点
                    ## 仍有效的矩阵remain_xy=xy_loss[:, delta_x: ]
                    new_loss = np.full((matrix_size, matrix_size), -1)
                    new_loss[: , : matrix_size - delta_x] = xy_loss[:, delta_x:]
                    xy_loss = new_loss

                if(delta_y < 0):
                    # 中心向上移动abs(delta_y)个坐标点, 矩阵向下移动abs(delta_y)个坐标点
                    ## 仍有效的矩阵remain_xy=xy_loss[: matrix_size + delta_y, : ]
                    new_loss = np.full((matrix_size, matrix_size), -1)
                    new_loss[abs(delta_y) : , : ] = xy_loss[: matrix_size + delta_y, : ]
                    xy_loss = new_loss
                elif(delta_y > 0):
                    # 中心向下移动abs(delta_y)个坐标点, 矩阵向上移动abs(delta_y)个坐标点
                    ## 仍有效的矩阵remain_xy=xy_loss[delta_y : , : ]
                    new_loss = np.full((matrix_size, matrix_size), -1)
                    new_loss[: matrix_size - delta_y, : ] = xy_loss[delta_y : , : ]
                    xy_loss = new_loss

                curr_x = next_x
                curr_y = next_y
                logging.debug("new_loss: " + str(xy_loss))

                # 判定xy complete
                ## 当xy同时抵达边界, 结束匹配
                if(abs(curr_x) == max_abs_x and abs(curr_y) == max_abs_y):
                    xy_complete = True
                ## 当矩阵未发生更新时结束匹配(此时矩阵无负值)
                if(xy_loss.min() >= 0):
                    xy_complete = True

                # 判定结束条件
                if(xy_changed):
                    # 如果xy发生移动, 则重新匹配angle offset
                    angle_complete = False
                    angle_changed = False
                else:
                    # 如果angle和xy都未发生移动, 则抵达最优结束匹配
                    if(angle_changed == False):
                        break
            elif(angle_complete == False):
                # 0. 检测是否需要进行旋转角匹配
                if(angle_accu <= 0):
                    ## 跳过匹配
                    angle_complete = True
                    angle_changed = False
                    continue
                # 1. 更新loss向量并寻找loss最小的angle值
                ## min_loss为本次更新后, 损失向量中最小的loss值
                min_loss = -1
                ## min_da为本次更新后, 损失向量中最小的loss值相对于中心的index偏移量
                ## 例如min_loss在中心偏左2个单位, 则min_da=-2
                min_da = 0
                for i in range(matrix_size):
                    if(angle_loss[i] < 0):
                        angle_loss[i] = self.try_match(
                            test, std, 
                            x=curr_x, 
                            y=curr_y, 
                            angle=curr_a + (i - view_size) * curr_angle_step,
                            shielded_areas=shielded_areas
                        )
                    if((min_loss < 0) or (angle_loss[i] < min_loss)):
                        min_loss = angle_loss[i]
                        min_da = i - view_size
                ## 当中心也为最小值时则不移动
                if(angle_loss[view_size] == min_loss):
                    min_da = 0

                # 2. 输出计算结果
                logging.debug("angle loss vector: %s"%(angle_loss))
                logging.debug("min_da: %d"%(min_da))
                logging.debug("min_loss: %d"%(min_loss))

                # 3. 计算目标旋转角
                _ = curr_a + min_da * curr_angle_step
                # 判定是否触发限幅, 如是则判定angle complete
                if(_ < - max_abs_a):
                    curr_a = - max_abs_a
                    angle_complete = True
                    angle_changed = True
                    continue
                elif(_ > max_abs_a):
                    curr_a = max_abs_a
                    angle_changed = True
                    angle_complete = True
                    continue
                
                # 4. 检测angle loss是否发生移动
                if(min_da != 0):
                    angle_changed = True

                # 5. 若未触发限幅, 向最小loss方向移动
                curr_a += min_da * curr_angle_step
                # 更新loss向量(用-1填充向量)
                if(min_da < 0):
                    # angle变小, 则向量向右移动, 并在左侧填充-1
                    new_loss = np.full(matrix_size, -1)
                    new_loss[abs(min_da):] = angle_loss[: matrix_size - abs(min_da)]
                    angle_loss = new_loss              
                elif(min_da > 0):
                    # angle变大, 则向量向左移动, 并在右侧填充-1
                    new_loss = np.full(matrix_size, -1)
                    new_loss[: matrix_size - abs(min_da)] = angle_loss[min_da:]
                    angle_loss = new_loss

                # 判定angle complete
                if(angle_loss.min() < 0):
                    ## 旋转角损失向量仍在更新, 则继续运算
                    pass
                else:
                    ## 旋转角损失向量未发生更新(此时矩阵无负值)
                    if(curr_angle_step >= angle_accu):
                        ## 旋转角损失向量未发生更新且未达到目标精度时, 继续提高精度
                        curr_angle_step /= 2.0
                        ## 清空损失向量, 并重新计算
                        new_loss = np.full(matrix_size, -1)
                        new_loss[view_size] = angle_loss[view_size]
                        angle_loss = new_loss
                    else:
                        ## 旋转角损失向量未发生更新且当前精度达到目标精度时, 结束运算
                        angle_complete = True
                        if(angle_changed):
                            # 如果angle发生移动, 则重新匹配xy offset
                            xy_complete = False
                            xy_changed = False
                        else:
                            # 如果angle和xy都未发生移动, 则抵达最优结束匹配
                            if(xy_changed == False):
                                break


            # 更新本代计算误差
            loss = self.try_match(
                test, std, 
                x=round(curr_x), 
                y=round(curr_y), 
                angle=curr_a,
                shielded_areas=shielded_areas,
                show_diff=show_process
            )


            if(loss < best_loss):
                best_loss = loss
                best_params = [curr_x, curr_y, curr_a]

            logging.info("iteration: %d, loss: %f, x: %d, y: %d, a: %f" % (iterations, loss, curr_x, curr_y, curr_a))

        return best_params



    def cut_with_tol(self, img1:np.ndarray, img2:np.ndarray, tolerance:int, shielded_areas:list=None):
        '''
        @brief: 膨胀相切算法, 用img1来切img2, 返回img2剩余部分, 使用时要交替互相相切最终得误差图
        @note: 图像均为二值图
        @param: 
            - img1
            - img2
            - tolerance: 像素误差值
        '''
        # 先进行膨胀相切(0腐蚀)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance * 2 + 1, tolerance * 2 + 1))
        img1_dilate = cv2.dilate(img1, kernel, 1)
        img2_dilate = cv2.dilate(img2, kernel, 1)
        xor = cv2.bitwise_xor(img1_dilate, img2_dilate)
        remain = cv2.bitwise_and(xor, img2)

        # 执行屏蔽
        if(isinstance(shielded_areas, list)):
            for area in shielded_areas:
                remain = cv2.rectangle(remain, (area[0], area[1]), (area[2], area[3]), 0, thickness=cv2.FILLED)
        return remain


    def match_template_to_target_partitioned(self, template_pattern:np.ndarray, target_pattern:np.ndarray, shielded_areas:list=None):
        """
        @brief: 将模板分区匹配到目标样式上, 并返回分区匹配后的结果
        @param:
            - template_pattern: 模板样式, 要求二值图
            - target_pattern: 待检图像样式, 要求二值图
            - shielded_areas: 屏蔽区域列表
        @return: 将模板分区匹配到待检图像后的结果
        """
        # 1. 先屏蔽待检区域
        if(isinstance(shielded_areas, list)):
            for area in shielded_areas:
                # 执行屏蔽
                template_pattern = cv2.rectangle(template_pattern, (area[0], area[1]), (area[2], area[3]), 0, thickness=cv2.FILLED)
                target_pattern  = cv2.rectangle(target_pattern, (area[0], area[1]), (area[2], area[3]), 0, thickness=cv2.FILLED)

        # 2. 做逻辑与操作, 寻找单个闭合区域
        closed_pattern = cv2.bitwise_or(template_pattern, target_pattern)

        # 3. 逐个遍历闭合区域, 并进行分区匹配
        result = np.zeros((target_pattern.shape[0], target_pattern.shape[1]), np.uint8)
        ## 3.1 寻找闭合区域时只检测外围轮廓
        contours, hierarchy = cv2.findContours(closed_pattern, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ## 3.2 分区域进行fine_tune
            template_partition = template_pattern[y : y + h, x : x + w]
            target_partition = target_pattern[y : y + h, x : x + w]
            logging.debug("shape: " + str(template_partition.shape))
            offset = self.fine_tune(
                test=template_partition,
                std=target_partition,
                max_abs_x=min(7, w),
                max_abs_y=min(7, h),
                max_abs_a=2.5,
                max_iterations=40,
                angle_accu=-1,
                view_size=10,
            )

            ## 3.3 将fine_tune结果拼回总图
            result[y : y + h, x : x + w] = self.linear_trans_to(
                img=template_partition,
                x=offset[0],
                y=offset[1],
                angle=offset[2],
                output_size=(w, h),
                border_color=0
            )

        return result


class InkDefectDetector():
    def __init__(self, path:str, img_sz:int) -> None:
        import onnxruntime as ort
        self._sesson = ort.InferenceSession(path)
        self._img_sz = img_sz


    def _postprocess(self, output, factor:float, conf_thres:float = 0.2, iou_thres:float = 0.2):
        """
        @brief: yolov8模型后处理
        @param: 
            - output: yolov8在onnx上的计算输出
            - factor: 图像的缩放系数(缩放时应保持横纵比)
        @return:
            [
                [x, y, w, h, confidence, cls]
            ]
        """
        # 转置并压缩输出以匹配期望的形状：(8400, 84)
        outputs = np.transpose(np.squeeze(output[0]))
        # 获取输出数组的行数
        rows = outputs.shape[0]
        # 存储检测到的边界框、分数和类别ID的列表
        boxes = []
        scores = []
        class_ids = []
        output = []

        # 遍历输出数组的每一行
        for i in range(rows):
            # 从当前行提取类别的得分
            classes_scores = outputs[i][4:]
            # 找到类别得分中的最大值
            max_score = np.amax(classes_scores)

            # 如果最大得分大于或等于置信度阈值
            if max_score >= conf_thres:
                # 获取得分最高的类别ID
                class_id = np.argmax(classes_scores)

                # 从当前行提取边界框坐标
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 计算边界框的缩放坐标
                x = round(x / factor)
                y = round(y / factor)
                w = round(w / factor)
                h = round(h / factor)

                # 将类别ID、得分和边界框坐标添加到相应的列表中
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([x, y, w, h])

        # 应用非极大抑制以过滤重叠的边界框
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

        # 遍历非极大抑制后选择的索引
        for i in indices:
            # 获取与索引对应的边界框、得分和类别ID
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            obj = box
            obj.append(score)
            obj.append(class_id)
            output.append(obj)

        return output


    def detect(self, img:np.ndarray, confidence_thre:float = 0.2, template_defects:list = []) -> list:
        """
        @brief: 检测待检图像的断墨缺陷
        @param:
            - img: opencv的BGR图像
            - template_defects: 
                - [x, y, w, h, confidence, class] 构成的列表
                - 模板图像的标定缺陷, 用于排除模板自带的缺陷, 以及追加模板自带但待测图像不带的缺陷
        @return: [x, y, w, h, confidence, class] 构成的列表
        """
        # 前处理
        ori_w = img.shape[1]
        ori_h = img.shape[0]
        scale = min(float(self._img_sz) / ori_w, float(self._img_sz) / ori_h)
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        border_h = self._img_sz - scaled_img.shape[0]
        border_w = self._img_sz - scaled_img.shape[1]
        bordered_img = cv2.copyMakeBorder(scaled_img, 0, border_h, 0, border_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = bordered_img[:, :, ::-1].transpose(2, 0, 1)
        img = img.astype(dtype=np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)

        # 推理
        input_name = self._sesson.get_inputs()[0].name
        output_names = [o.name for o in self._sesson.get_outputs()]
        output = self._sesson.run(output_names, {input_name: img})[0]

        # 模型后处理
        defects = self._postprocess(output, scale, confidence_thre)

        # 和模板进行比对
        if(len(template_defects)):
            # 为defects补充template中有但是img中没有的缺陷
            img_lost = []
            for defect in template_defects:
                x, y, w, h, confidence, cls = defect
                lost = True
                for out in defects:
                    out_x, out_y, out_w, out_h, out_conf, out_cls = out
                    if(
                        x <= out_x + out_w / 2 and
                        x >= out_x - out_w / 2 and
                        y <= out_y + out_h / 2 and
                        y >= out_y - out_h / 2
                    ):
                        # 未丢失
                        lost = False
                        break
                if(lost):
                    img_lost.append(out)

            # 忽略defects中与template中重复的元素
            output = img_lost
            for defect in defects:
                x, y, w, h, confidence, cls = defect
                duplicate = False
                for temp_defect in template_defects:
                    tem_x, tem_y, tem_w, tem_h, tem_conf, tem_cls = temp_defect
                    if(
                        x <= tem_x + tem_w / 2 and
                        x >= tem_x - tem_w / 2 and
                        y <= tem_y + tem_h / 2 and
                        y >= tem_y - tem_h / 2
                    ):
                        # 发生重复
                        duplicate = True
                        break
                if(not duplicate):
                    output.append(defect)

            return output
        else:
            return defects
    