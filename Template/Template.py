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


    def get_shielded_areas(self):
        return self._configs["shielded_areas"]


    def get_img_path(self):
        return os.path.join(self._save_path, self._name + "." + self._configs["img_type"])


    def set_name(self, name:str):
        self._name = name
    

    def set_img_type(self, type:str):
        self._configs["img_type"] = type


    def add_shielded_area(self, x1:int, y1:int, x2:int, y2:int):
        """向Template中添加一个屏蔽区域"""
        area = {}
        area["x1"] = x1
        area["y1"] = y1
        area["x2"] = x2
        area["y2"] = y2
        area_list = self._configs["shielded_areas"]
        area_list.append(area)
        self._configs["shielded_areas"] = area_list


    def save(self):
        """
        将Template示例保存到指定路径中

        Note: 该方法可能会抛出异常, 注意处理。
        """
        # 检查变量
        self._check_configs()
        with open(self._yml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data=self._configs, stream=f, allow_unicode=True)

