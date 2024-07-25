import numpy as np
import cv2
import math

'''
@brief: find label in img
@note: The image is the smallest possible image containing labels.
@return: minAreaRect
'''
def find_label(img, ed_kernel_size = 3, ed_iterations = 3, show_soble = False, show_dilate = False):
    # 提取标准模板所在色块
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_soble_x = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize = 3)
    img_soble_y = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize = 3)
    img_soble = cv2.addWeighted(img_soble_x, 0.5, img_soble_y, 0.5, 0)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ed_kernel_size, ed_kernel_size))
    img_aera_erode = cv2.erode(img_soble, erode_kernel, ed_iterations)
    img_dilate = cv2.dilate(img_aera_erode, erode_kernel, ed_iterations)

    # 寻找最大区域
    _, contours, hierarchy1 = cv2.findContours(img_dilate, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    max_countour = None
    max_size = -1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if(w * h > max_size):
            max_size = w * h
            max_countour = c

    # 绘制最大区域
    min_area_rect = None
    if(max_countour is not None):
        min_area_rect = cv2.minAreaRect(max_countour)
    
    # 绘制中间结果
    if(show_soble):
        cv2.imshow("img_soble", img_soble)
    if(show_dilate):
        cv2.imshow("img_dilate", img_dilate)
    return min_area_rect


'''
@note: This function will check if the rotation angle is less than 30 degrees
'''
def check_min_aera_rect(min_area_rect) -> bool:
    angle = min_area_rect[2]
    if(abs(angle) > 30 and abs(angle) < 60):
        return False
    else:
        return True


def wrap_min_aera_rect(src, min_area_rect):
    print(min_area_rect)
    # 修正参数
    _w = min_area_rect[1][0]
    _h = min_area_rect[1][1]
    print(_w)
    print(_h)
    if(_w < _h):
        min_area_rect = (
            min_area_rect[0], 
            (min_area_rect[1][1], min_area_rect[1][0]), 
            min_area_rect[2] + 90
        )

    # 计算仿射目标点
    x = min_area_rect[0][0]
    y = min_area_rect[0][1]
    w = min_area_rect[1][0]
    h = min_area_rect[1][1]
    
    # 计算仿射前四个顶点位置
    points = cv2.boxPoints(min_area_rect)
    src_array = np.array(
        [
            # 左上、右上
            points[1], points[2],
            # 左下、右下
            points[0], points[3],
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
    print(src_array)
    print(dst_array)
    matrix = cv2.getPerspectiveTransform(src_array, dst_array)
    return cv2.warpPerspective(src, matrix, (int(w), int(h)))


'''
@brief: 将图像进行线性变换, 边缘用黑色填充
@param:
    - output_size: 输出图像大小, 默认为原大小, 保持中心对齐。大则裁切小则填充。
        output_size[0]: w
        output_size[1]: h
@return:
    线性变换并保持中心对齐后, 经过裁切或填充的图像。
'''
def linear_trans_to(img, x:int, y:int, angle:float, output_size = [-1, -1], border_color = [0, 0, 0]):
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
@brief: 将待测图像以指定线性变换匹配到标准图像上
@return: 像素差
'''
def try_match(test, std, x:int, y:int, angle, show_diff = False) -> int:
    trans = linear_trans_to(test, x, y, angle, [std.shape[1], std.shape[0]], 0)
    # 计算像素点误差
    diff = cv2.absdiff(trans, std)
    if(show_diff):
        cv2.imshow("try_match:diff", diff)
        cv2.imwrite("../try_match.diff.jpg", diff)
    loss = cv2.countNonZero(diff)
    return loss


'''
@param:
    - view_size: 视野边距: 
        例如边距为为1时, 对应视野矩阵为3x3, 视野行向量长度为3
        例如边距为为2时, 对应视野矩阵为5x5, 视野行向量长度为5
@return:
    best param in [x, y, angle]
@note: 本函数只能处理微小误差(半行以内)
'''
def match(test, std, max_abs_x:int, max_abs_y:int, max_abs_a:float, max_iterations, angle_step = 0.1, view_size:int = 2):
    iterations = 0
    # 初始值与现值
    curr_x = 0
    curr_y = 0
    curr_a = 0
    ## [x, y, angle]
    last_params = [0, 0, 0.]
    last_loss = try_match(test, std, 0, 0, 0)
    best_params = last_params
    best_loss = last_loss
    print("iteration: %d, loss: %f" % (iterations, last_loss))

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
                        xy_loss[i][j] = try_match(test, std, curr_x + j -1, curr_y + i - 1, curr_a)
                    if((min_loss < 0) or (xy_loss[i][j] < min_loss)):
                        min_loss = xy_loss[i][j]
                        min_dxy = [ j - 1, i - 1 ]
            
            print(xy_loss)
            print("minxy: ", min_dxy)
            print("minloss: %d"%(min_loss))

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
            print(xy_loss)
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
                    angle_loss[i] = try_match(test, std, curr_x, curr_y, curr_a + (i - view_size) * angle_step)
                if((min_loss < 0) or (angle_loss[i] < min_loss)):
                    min_loss = angle_loss[i]
                    min_da = i - view_size

            # 当中心也为最小值时则不移动
            if(angle_loss[view_size] == min_loss):
                min_da = 0

            print(angle_loss)
            print("min_da: %d"%(min_da))
            print("min_loss: %d"%(min_loss))

            
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
                angle_loss = np.full(abs(min_da), -1).extend(angle_loss[:-abs(offset)])               
            elif(min_da > 0):
                # angle变大，则向量向左移动
                angle_loss = angle_loss[:-abs(min_da)].extend(np.full(abs(offset), -1))

            print(angle_loss)

            # 判定angle complete
            ## 当矩阵未发生更新时结束匹配(此时矩阵无负值)
            if(angle_loss.min() >= 0):
                angle_complete = True

        else:
            # Finish
            break

        # 更新本代计算误差
        loss = try_match(test, std, round(curr_x), round(curr_y), curr_a, show_diff=True)

        if(loss < best_loss):
            best_loss = loss
            best_params = [curr_x, curr_y, curr_a]

        print("iteration: %d, loss: %f, x: %d, y: %d, a: %f" % (iterations, loss, curr_x, curr_y, curr_a))
        print(cv2.waitKey(1000))

    return best_params


'''
@brief: 用img1来切img2, 返回img2剩余部分
@note: 图像均为二值图
@param: 
    - tolerance: 像素误差值
'''
def cut_with_tol(img1, img2, tolerance:int):
    # 将像素进行膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tolerance * 2 + 1, tolerance * 2 + 1))
    img1_dilate = cv2.dilate(img1, kernel, 1)
    img2_dilate = cv2.dilate(img2, kernel, 1)

    xor = cv2.bitwise_xor(img1_dilate, img2_dilate)
    remain = cv2.bitwise_and(xor, img2)
    return remain



if __name__ == '__main__':
    
    #std = cv2.imread("../std.bmp")
    std = cv2.imread("../std1.jpg")
    #test = cv2.imread("../detect.bmp")
    test = cv2.imread("../detect1.jpg")

    cv2.imshow("std", std)
    cv2.imshow("test", test)

    std_min_area_rect = find_label(std, show_soble=False, show_dilate=False)
    test_min_area_rect = find_label(test, show_soble=False, show_dilate=False)

    # 检查检测结果，TODO
    check_min_aera_rect(std_min_area_rect)
    check_min_aera_rect(test_min_area_rect)

    # 绘制识别区域
    draw_min_rect = False
    if(draw_min_rect):
        cv2.imshow("std_rect", cv2.resize(cv2.drawContours(std, [np.int0(cv2.boxPoints(std_min_area_rect))], 0, (0, 255, 0), 1), (0, 0), fx=0.5, fy=0.5))
        cv2.imshow("test_rect", cv2.resize(cv2.drawContours(test, [np.int0(cv2.boxPoints(test_min_area_rect))], 0, (0, 255, 0), 1), (0, 0), fx=0.5, fy=0.5))

    std_wraped = wrap_min_aera_rect(std, std_min_area_rect)
    test_wraped = wrap_min_aera_rect(test, test_min_area_rect)

    #cv2.imshow("std_wraped", std_wraped)
    #cv2.imshow("test_wraped", test_wraped)
    

    std_wraped_reversed = cv2.bitwise_not(std_wraped)
    test_wraped_reversed = cv2.bitwise_not(test_wraped)

    _, std_pattern = cv2.threshold(cv2.cvtColor(std_wraped_reversed, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)
    _, test_pattern = cv2.threshold(cv2.cvtColor(test_wraped_reversed, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("std_pattern", std_pattern)
    cv2.imshow("test_pattern", test_pattern)

    cv2.imwrite("./output/std_pattern.jpg", std_pattern)
    cv2.imwrite("./output/test_pattern.jpg", test_pattern)

    # 初步判定图像相似度
    loss = try_match(test_pattern, std_pattern, 0, 0, 0, show_diff = True)
    # 当图像相似度较低时跳过match, 直接判定问题 TODO

    # 使用梯度下降进一步调整仿射角度, TODO 角度step
    [x, y, angle] = match(test_pattern, std_pattern, 20, 20, 3.0, 100, 0.05 / loss)
    # 对待测目标进行仿射变换
    test_final = linear_trans_to(test_pattern, x, y, angle, [std_pattern.shape[1], std_pattern.shape[0]])
    cv2.imshow("test_final", test_final)

    test_remain = cut_with_tol(std_pattern, test_final, 3)
    cv2.imshow("test_remain", test_remain)
    std_remain = cut_with_tol(test_final, std_pattern, 3)
    cv2.imshow("std_remain", std_remain)
    #try_match(test_pattern, std_pattern, 0, 0, 0, show_diff = True)



    cv2.waitKey(0)
    exit()

