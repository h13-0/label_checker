import os
import yaml
import logging

class Template():
    def __init__(self, save_path:str) -> None:
        # 基本变量
        ## 模板保存位置
        self._save_path = save_path
        ## 模板名称
        self._name = os.path.basename(save_path)
        self._yml_path = os.path.join(save_path, self._name + ".yml")
        self._configs = {}


    def __str__(self) -> str:
        s = "type: Template"
        s += ", name: " + str(self._name)
        s += ", yml: " + str(self._yml_path)
        s += ", img: " + str(self.get_img_path())
        s += ", cfg: " + str(self._configs)
        return s


    def _check_configs(self):
        """检查self._configs字典中的配置文件是否合法"""
        # 校验图片类型
        if(
            not isinstance(self._configs["img_type"], str) or
            len(self._configs["img_type"]) == 0
        ):
            msg = "\"img_type\" field configuration error."
            logging.error(msg)
            raise RuntimeError(msg)

        ## 检查图片文件是否存在
        img_path = self.get_img_path()
        if(
            not os.path.exists(img_path) or
            not os.path.isfile(img_path)
        ):
            msg = "img file: " + img_path + " not exists."
            logging.error(msg)
            raise RuntimeError(msg)
    
        ## 将config中的坐标自动转为int
        if("shielded_areas" in self._configs):
            for area in self._configs["shielded_areas"]:
                area["x1"] = round(area["x1"])
                area["y1"] = round(area["y1"])
                area["x2"] = round(area["x2"])
                area["y2"] = round(area["y2"])


    @staticmethod
    def open(save_path:str):
        """
        静态构造方式, 直接通过打开保存路径进行构造

        Note: 该方法可能会抛出异常, 注意处理。
        """
        template = Template(save_path)
        # 检查文件夹是否存在
        if(not os.path.isdir(save_path)):
            msg = "path: " + save_path + " is not a folder"
            logging.error(msg)
            raise RuntimeError(msg)
        
        # 检查文件是否存在
        if(
            not os.path.exists(template._yml_path) or
            not os.path.isfile(template._yml_path)
        ):
            msg = "config file: " + template._yml_path + " not exists."
            logging.error(msg)
            raise RuntimeError(msg)


        # 读取配置文件
        with open(template._yml_path, 'r', encoding='utf-8') as f:
            template._configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 校验配置文件
        template._check_configs()

        return template


    def get_img_path(self):
        return os.path.join(self._save_path, self._name + "." + self._configs["img_type"])


    def set_name(self, name:str):
        self._name = name
    

    def set_img_type(self, type:str):
        self._configs["img_type"] = type


    def get_shielded_areas(self):
        return self._configs.get("shielded_areas", [])
    

    def add_shielded_area(self, x1:int, y1:int, x2:int, y2:int):
        """向Template中添加一个屏蔽区域"""
        area = {}
        area["x1"] = x1
        area["y1"] = y1
        area["x2"] = x2
        area["y2"] = y2
        area_list = self._configs.get("shielded_areas", [])
        area_list.append(area)
        self._configs["shielded_areas"] = area_list


    def get_hsv_threshold(self) -> dict:
        """
        @brief: 获取当前模板的HSV阈值
        @return: dict with:
            {
                "h_min": h_min,
                "h_max": h_max,
                "s_min": s_min,
                "s_max": s_max,
                "v_min": v_min,
                "v_max": v_max,
            }
        """
        return self._configs.get(
            "hsv_threshold", 
            {
                "h_min": 0,
                "h_max": 255,
                "s_min": 0,
                "s_max": 255,
                "v_min": 0,
                "v_max": 255,
            }
        )


    def set_hsv_threshold(self,
            h_min:int, h_max:int,
            s_min:int, s_max:int,
            v_min:int, v_max:int
        ):
        hsv_thre = {}
        hsv_thre["h_min"] = h_min
        hsv_thre["h_max"] = h_max
        hsv_thre["s_min"] = s_min
        hsv_thre["s_max"] = s_max
        hsv_thre["v_min"] = v_min
        hsv_thre["v_max"] = v_max
        self._configs["hsv_threshold"] = hsv_thre


    def get_depth_threshold(self) -> int:
        return self._configs.get("depth_threshold", 0)


    def set_depth_threshold(self, threshold:int):
        self._configs["depth_threshold"] = threshold


    def save(self):
        """
        将Template示例保存到指定路径中

        Note: 该方法可能会抛出异常, 注意处理。
        """
        # 检查变量
        self._check_configs()
        with open(self._yml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=self._configs, stream=f, allow_unicode=True)

