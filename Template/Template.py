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
        self._default_configs = {
            "bartender_path": "",
            "img_sample_path": "",
            "hsv_threshold": {
                "h_max": 255,
                "h_min": 0,
                "s_max": 255,
                "s_min": 0,
                "v_max": 255,
                "v_min": 0,
            },
            "depth_threshold": 126,
            "barcode_sources": {},
            "ocr_sources": {}
        }


    def __str__(self) -> str:
        s = "type: Template"
        s += ", name: " + str(self._name)
        s += ", yml: " + str(self._yml_path)
        s += ", cfg: " + str(self._configs)
        return s


    def _check_configs(self, configs:dict):
        """
        @brief: 检查self._configs字典中的配置文件是否合法
        @return: bool
        """
        for key in self._default_configs:
            # 检查key是否存在
            if(not key in configs):
                logging.error("key :" + key + " not configured.")
                return False
            
            # 检查类型是否正确
            if(not isinstance(configs[key], type(self._default_configs[key]))):
                logging.error("the type of key: " + key + " is incorrect.")
                logging.error("current type of key: " + key + " is " + str(type(configs[key])))
                return False

            # 检查特殊类
            if(key == "bartender_path" or key == "img_sample_path"):
                if(not os.path.exists(configs[key])):
                    logging.error("file: " + configs[key] + " doesn't exist.")
                    return False

            elif(key == "hsv_threshold"):
                if(not set(self._default_configs[key].keys()).issubset(set(configs[key].keys()))):
                    logging.error("the key values in hsv_threshold are incomplete.")
                    return False
                for i in configs[key]:
                    configs[key][i] = round(configs[key][i])
                    
            elif(key == "barcode_sources" or key == "ocr_sources"):
                keys_to_check = { "x1", "y1", "x2", "y2" }
                for id in configs[key]:
                    if(not keys_to_check.issubset(set(configs[key][id].keys()))):
                        logging.error("the key values in " + key + "." + id + " are incomplete.")
                        return False
                    for coor in configs[key][id]:
                        configs[key][id][coor] = round(configs[key][id][coor])
        return True


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
            return None
        
        # 检查文件是否存在
        if(
            not os.path.exists(template._yml_path) or
            not os.path.isfile(template._yml_path)
        ):
            msg = "config file: " + template._yml_path + " not exists."
            logging.error(msg)
            return None


        # 读取配置文件
        with open(template._yml_path, 'r', encoding='utf-8') as f:
            template._configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 校验配置文件
        if(not template._check_configs(template._configs)):
            return None

        return template


    def get_img_sample_path(self):
        """
        @brief: 获取样本图像路径
        """
        return self._configs["img_sample_path"]


    def get_bartender_template_path(self):
        """
        @brief: 获取BarTender的模板路径
        """
        return self._configs["bartender_path"]


    def set_name(self, name:str):
        self._name = name


    def get_shielded_areas(self):
        return self._configs.get("shielded_areas", [])


    def add_barcode_source(self, id:str, x1:int, y1:int, x2:int, y2:int):
        """
        @brief: 向Template中添加一个条码数据源
        @param:
            - id: 数据源名称
            - x1: 左上角x轴坐标
            - y1: 左上角y轴坐标
            - x2: 右下角x轴坐标
            - y2: 右下角y轴坐标
        """
        barcode_source = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        self._configs["barcode_sources"][id] = barcode_source


    def add_ocr_source(self, id:str, x1:int, y1:int, x2:int, y2:int):
        """
        @brief: 向Template中添加一个OCR数据源
        @param:
            - id: 数据源名称
            - x1: 左上角x轴坐标
            - y1: 左上角y轴坐标
            - x2: 右下角x轴坐标
            - y2: 右下角y轴坐标
        """
        ocr_source = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
        self._configs["ocr_sources"][id] = ocr_source


    def delete_barcode_source(self, id:str):
        self._configs["barcode_sources"].pop(id)


    def delete_ocr_source(self, id:str):
        self._configs["ocr_sources"].pop(id)


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
        hsv_thre = {
            "h_min": h_min,
            "h_max": h_max,
            "s_min": s_min,
            "s_max": s_max,
            "v_min": v_min,
            "v_max": v_max
        }
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
        if(not self._check_configs(self._configs)):
            raise RuntimeError("template configure failed.")
        with open(self._yml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=self._configs, stream=f, allow_unicode=True)

