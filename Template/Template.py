import os
import yaml
import logging

class Template():
    def __init__(self) -> None:
        # 基本变量
        ## 模板名称
        self._name = None
        self._yml_path = None
        self._img_path = None
        self._configs = None


    def __str__(self) -> str:
        s = "type: Template"
        s += ", name: " + str(self._name)
        s += ", yml: " + str(self._yml_path)
        s += ", img: " + str(self._img_path)
        s += ", cfg: " + str(self._configs)
        return s


    @staticmethod
    def open(folder_path:str):
        template = Template()
        # 检查文件夹是否存在
        if(not os.path.isdir(folder_path)):
            msg = "path: " + folder_path + " is not a folder"
            logging.error(msg)
            raise RuntimeError(msg)
        
        # 填装基本变量
        template._name = os.path.basename(folder_path)
        template._yml_path = os.path.join(folder_path, template._name + ".yml")
        
        # 检查文件是否存在
        if(
            not os.path.exists(template._yml_path) or
            not os.path.isfile(template._yml_path)
        ):
            msg = "config file: " + template._yml_path + "not exists."
            logging.error(msg)
            raise RuntimeError(msg)


        # 读取配置文件
        with open(template._yml_path, 'r', encoding='utf-8') as f:
            template._configs = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 校验配置文件
        ## 校验图片类型
        if(
            not isinstance(template._configs["img_type"], str) or
            len(template._configs["img_type"]) == 0
        ):
            msg = "\"img_type\" field configuration error."
            logging.error(msg)
            raise RuntimeError(msg)

        ## 检查文件是否存在
        template._img_path = os.path.join(folder_path, template._name + "." + template._configs["img_type"])
        if(
            not os.path.exists(template._img_path) or
            not os.path.isfile(template._img_path)
        ):
            msg = "img file: " + template._img_path + " not exists."
            logging.error(msg)
            raise RuntimeError(msg)
        return template

    def get_shielded_areas(self):
        return self._configs["shielded_areas"]


    def get_img_path(self):
        return self._img_path



    def save(self):
        pass
