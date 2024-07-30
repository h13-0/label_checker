import numpy as np
import cv2
import math
import logging

class LabelChecker():
    def __init__(self) -> None:
        pass


    '''
    @brief: 从图像中寻找标签
    @note: 输入图像应当仅包含一个标签
    @return: minAreaRect
    '''
    def find_label(self, 
        img, h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int,
        show_mask:bool = False
    ):
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
            size = minRect[1][0] * minRect[1][1]
            if(size > max_size):
                max_size = size
                max_area_rect = minRect

        return max_area_rect


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
    def find_labels(self, 
        img, template_w:int, template_h:int,
        h_min:int, h_max:int, s_min:int, s_max:int, v_min:int, v_max:int,
        wh_tol_ratio:float = 0.1, show_mask:bool = False
    ):
        # 使用HSV提取标签所在色块
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        if(show_mask):
            cv2.imshow("@find_label:mask", mask)

        # 寻找标签所在区域
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        rects = []
        for c in contours:
            minRect  = cv2.minAreaRect(c)
            w = max(minRect[1][0], minRect[1][1])
            h = min(minRect[1][0], minRect[1][1])

            if(
                abs((w / float(template_w)) - 1) < wh_tol_ratio and
                abs((h / float(template_h)) - 1) < wh_tol_ratio
            ):
                rects.append(cv2.minAreaRect(c))

        return rects


    '''
    @brief: 该函数主要用于检测标签的 minAreaRect 倾角是否小于30
    '''
    def check_min_aera_rect(self, min_area_rect) -> bool:
        angle = min_area_rect[2]
        if(abs(angle) > 30 and abs(angle) < 60):
            return False
        else:
            return True


    """
    @brief: 将min_area_rect转换为四个顶点, 横方向为长边并且输出顺序为: [左上 右上 左下 右下],
    @return:
        - 横方向为长边并且输出list为[左上 右上 左下 右下]
    """
    def _get_box_point(self, min_area_rect) -> list:
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


    '''
    @brief: 将 minAreaRect 所在区域仿射并裁切到长方形
    @note: 该函数会自动将图像旋转到正确的方向
    '''
    def wrap_min_aera_rect(self, src, min_area_rect):
        # 提取w、h
        w = min_area_rect[1][0]
        h = min_area_rect[1][1]
        if(w < h):
            w = min_area_rect[1][1]
            h = min_area_rect[1][0]

        # 计算仿射前四个顶点位置
        points = self._get_box_point(min_area_rect)
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


    """
    @brief: 将经过仿射变换的标签图像转化为样式二值图
    @param:
        - wraped_img: 仿射后的图像
        - threshold: 黑度阈值(255: 最黑, 0: 最亮)
    """
    def get_pattern(self, wraped_img, threshold:int):
        # 1. 将图像反色
        target_wraped_reversed = cv2.bitwise_not(wraped_img)

        # 2. 将标签图像转为二值图
        _, target_pattern = cv2.threshold(cv2.cvtColor(target_wraped_reversed, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)

        return target_pattern


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
    def linear_trans_to(self, img, x:int, y:int, angle:float, output_size:list = [-1, -1], border_color = [0, 0, 0]):
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
        border_w = math.ceil((ori_hypo - ori_w) / 2) + x
        border_h = math.ceil((ori_hypo - ori_h) / 2) + y
        test_ext = cv2.copyMakeBorder(img, border_h, border_h, border_w, border_w, cv2.BORDER_CONSTANT, value=border_color)
        ext_w = test_ext.shape[1]
        ext_h = test_ext.shape[0]

        # 对test图像进行旋转
        matrix = cv2.getRotationMatrix2D(center=(ext_w / 2, ext_h / 2), angle=angle, scale=1)
        test_rot = cv2.warpAffine(test_ext, matrix, (ext_w, ext_h))

        # 从中心裁切到与输出大小同大，并在裁切时位移
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


    '''
    @brief: 将待测图像以指定线性变换匹配到标准图像上, 用于二值图匹配
    @param:
        - img1
        - img2
    @return: 未能匹配的像素数量
    '''
    def try_match(self, img1, img2, x:int, y:int, angle, show_diff = False) -> int:
        trans = self.linear_trans_to(img1, x, y, angle, [img2.shape[1], img2.shape[0]], 0)
        # 计算像素点误差
        diff = cv2.absdiff(trans, img2)
        if(show_diff):
            cv2.imshow("try_match:diff", diff)
        loss = cv2.countNonZero(diff)
        return loss


    '''
    @brief: 将待测图像(test)通过线性变换微调到模板(std)上。
    @param:
        - test
        - std
        - max_abs_x: x轴最大微调像素数(绝对值)
        - max_abs_y: y轴最大微调像素数(绝对值)
        - max_abs_a: 最大旋转角角度数(绝对值)
        - max_iterations: 最大微调迭代数
        - angle_step: 每步角度微调幅度
        - view_size: 视野边距: 
            例如边距为为1时, 对应视野矩阵为3x3, 视野行向量长度为3
            例如边距为为2时, 对应视野矩阵为5x5, 视野行向量长度为5
    @return:
        best param in [x, y, angle]
    @note: 本函数只能处理微小误差(半行以内), 即初始时应当已几乎匹配
    '''
    def fine_tune(self, test, std, max_abs_x:int, max_abs_y:int, max_abs_a:float, max_iterations, angle_step = 0.1, view_size:int = 2):
        iterations = 0
        # 初始值与现值
        curr_x = 0
        curr_y = 0
        curr_a = 0
        ## [x, y, angle]
        last_params = [0, 0, 0.]
        last_loss = self.try_match(test, std, 0, 0, 0)
        best_params = last_params
        best_loss = last_loss
        logging.info("iteration: %d, loss: %f" % (iterations, last_loss))

        # xy损失矩阵，未计算点位为-1，TODO: 可变视野大小
        xy_loss = np.full((3, 3), -1)
        # 旋转角度损失矩阵，未计算点为-1
        angle_loss = np.full(view_size * 2 + 1, -1)

        # 定义结束变量
        xy_complete = False
        angle_complete = False
        while(iterations < max_iterations):
            iterations += 1
            
            if(xy_complete == False):

                # 更新loss矩阵，并寻找loss最低值
                min_dxy = [0, 0]
                min_loss = -1
                for i in range(3):
                    for j in range(3):
                        if(xy_loss[i][j] < 0):
                            xy_loss[i][j] = self.try_match(test, std, curr_x + j -1, curr_y + i - 1, curr_a)
                        if((min_loss < 0) or (xy_loss[i][j] < min_loss)):
                            min_loss = xy_loss[i][j]
                            min_dxy = [ j - 1, i - 1 ]
                
                logging.info("xy_loss matrix: %s"%(str(xy_loss)))
                logging.info("minxy: %s"%(str(min_dxy)))
                logging.info("minloss: %d"%(min_loss))

                # 当中心也为最小值时则不移动，同时停止匹配
                if(xy_loss[1][1] == min_loss):
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
                
                # 将x、y指向loss更低的位置，并更新loss矩阵(用-1填充矩阵)
                if(next_x < curr_x):
                    # 中心向左移动，矩阵向右移动
                    xy_loss = np.concatenate((np.full((3, 1), -1), xy_loss[:, :-1]), axis = 1)                
                elif(next_x > curr_x):
                    # 中心向右移动，矩阵向左移动
                    xy_loss = np.concatenate((xy_loss[:, 1:], np.full((3, 1), -1)), axis = 1)
                if(next_y < curr_y):
                    # 中心向上移动，矩阵向下移动
                    xy_loss = np.concatenate((np.full((1, 3), -1), xy_loss[:-1]), axis = 0)
                elif(next_y > curr_y):
                    # 中心向下移动，矩阵向上移动
                    xy_loss = np.concatenate((xy_loss[1:], np.full((1, 3), -1)), axis = 0)

                curr_x = next_x
                curr_y = next_y

                # 判定xy complete
                ## 当xy同时抵达边界，结束匹配
                if(abs(curr_x) == max_abs_x and abs(curr_y) == max_abs_y):
                    xy_complete = True
                ## 当矩阵未发生更新时结束匹配(此时矩阵无负值)
                if(xy_loss.min() >= 0):
                    xy_complete = True
            elif(angle_complete == False):
                # 更新loss向量并寻找loss最小的angle值
                min_da = 0
                min_loss = -1
                for i in range(view_size * 2 + 1):
                    if(angle_loss[i] < 0):
                        angle_loss[i] = self.try_match(test, std, curr_x, curr_y, curr_a + (i - view_size) * angle_step)
                    if((min_loss < 0) or (angle_loss[i] < min_loss)):
                        min_loss = angle_loss[i]
                        min_da = i - view_size

                # 当中心也为最小值时则不移动
                if(angle_loss[view_size] == min_loss):
                    min_da = 0

                logging.info("angle loss vector: %s"%(angle_loss))
                logging.info("min_da: %d"%(min_da))
                logging.info("min_loss: %d"%(min_loss))

                
                _ = curr_a + min_da * angle_step
                # 判定是否触发限幅，如是则判定angle complete
                if(_ < - max_abs_a):
                    curr_a = - max_abs_a
                    angle_complete = True
                    continue
                elif(_ > max_abs_a):
                    curr_a = max_abs_a
                    angle_complete = True
                    continue
                
                # 若未限幅，向最小loss方向移动
                curr_a += min_da * angle_step
                # 更新loss向量(用-1填充向量)
                if(min_da < 0):
                    # angle变小，则向量向右移动
                    angle_loss = np.full(abs(min_da), -1).extend(angle_loss[:-abs(min_da)])               
                elif(min_da > 0):
                    # angle变大，则向量向左移动
                    angle_loss = angle_loss[:-abs(min_da)].extend(np.full(abs(min_da), -1))

                # 判定angle complete
                ## 当矩阵未发生更新时结束匹配(此时矩阵无负值)
                if(angle_loss.min() >= 0):
                    angle_complete = True

            else:
                # Finish
                break

            # 更新本代计算误差
            loss = self.try_match(test, std, round(curr_x), round(curr_y), curr_a, show_diff=False)

            if(loss < best_loss):
                best_loss = loss
                best_params = [curr_x, curr_y, curr_a]

            logging.info("iteration: %d, loss: %f, x: %d, y: %d, a: %f" % (iterations, loss, curr_x, curr_y, curr_a))

        return best_params


    '''
    @brief: 用img1来切img2, 返回img2剩余部分
    @note: 图像均为二值图
    @param: 
        - img1
        - img2
        - tolerance: 像素误差值
    '''
    def cut_with_tol(self, img1, img2, tolerance:int):
        # 将像素进行膨胀
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance * 2 + 1, tolerance * 2 + 1))
        img1_dilate = cv2.dilate(img1, kernel, 1)
        img2_dilate = cv2.dilate(img2, kernel, 1)

        xor = cv2.bitwise_xor(img1_dilate, img2_dilate)
        remain = cv2.bitwise_and(xor, img2)
        return remain
