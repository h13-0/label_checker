import clr
clr.AddReference("Seagull.BarTender.Print")
from Seagull.BarTender.Print import Engine, Printers, LabelFormatDocument, ImageType, ColorDepth, Resolution, OverwriteOptions
import logging

class BarTender():
    def __init__(self):
        self.btEngine = Engine(True)

    def get_printer_name_list(self) -> list:
        """
        @brief: 获取打印机列表, 返回值为打印机名构成的列表
        """
        printers = Printers()
        printer_list = []
        for printer in printers:
            printer_list.append(printer.PrinterName)
        return printer_list


    def gen_format(self, template_path:str, data_dict:dict) -> LabelFormatDocument:
        """
        @brief: 根据data_dict中指定的数据源信息创建打印样式
        @param: 
            - template_path: BarTender的模板文件路径
            - data_dict: 由Key-Value构成的打印样式
        """
        format = self.btEngine.Documents.Open(template_path)
        if len(data_dict) and format:
            for key, value in data_dict.items():
                for substring in format.SubStrings:
                    if substring.Name == key:
                        format.SubStrings.SetSubString(key, value)
        return format


    def export_format_to_img_file(self, 
        format:LabelFormatDocument, path:str, img_type:ImageType, 
        color_depth:ColorDepth, res:Resolution
    ):
        format.ExportImageToFile(
            fileName=path, 
            imageType=img_type,
            colorDepth=color_depth,
            resolution=res,
            overwriteOptions=OverwriteOptions.Overwrite
        )


    def create_print_task(self, format:LabelFormatDocument, printer_name:str):
        format.PrintSetup.PrinterName = printer_name
        format.Print()


    def __del__(self):
        if self.btEngine.IsAlive:
            self.btEngine.Stop()


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.DEBUG,
        #filename="log/"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log",
        #filemode="a"
    )

    dict = {
        "SN" : "B2108000033",
        "IMEI": "869128060787361"
    }

    tender = BarTender()
    
    format = tender.gen_format(r"D:\Projects\label_checker\Software\label_checker\BarTender\eye.btw", dict)

    tender.export_format_to_img_file(
        format=format,
        path="./1.jpg",
        img_type=ImageType.JPEG,
        color_depth=ColorDepth.ColorDepth24bit,
        res=Resolution(1080, 1080)
    )
