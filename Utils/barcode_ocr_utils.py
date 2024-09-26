from pyzbar.pyzbar import decode
from PIL import Image
import easyocr

class BarcodeRecognizer:
    def __init__(self):
        pass

    def recognize_barcode(self, image_path):
        result = decode(Image.open(image_path))
        if result:
            barcode_data = result[0].data.decode('utf-8')
            return barcode_data
        else:
            return None

class OCRRecognizer:
    def __init__(self, language='en'):
        self.reader = easyocr.Reader([language])

    def recognize_ocr(self, image_path):
        result = self.reader.readtext(image_path)
        extracted_text = [detection[1] for detection in result]
        return extracted_text

    def crop_image(self, image_path, coordinates):
        image = Image.open(image_path)
        cropped_image = image.crop(coordinates)
        return cropped_image

# 示例用法
barcode_recognizer = BarcodeRecognizer()
ocr_recognizer = OCRRecognizer()

# 通过条形码识别
barcode_result = barcode_recognizer.recognize_barcode('D:/software/label_checker-for-BarTender/test_img/SN_barcode.png')
if barcode_result:
    print("Barcode data:", barcode_result)
else:
    print("No barcode detected.")

# 通过OCR识别
ocr_result = ocr_recognizer.recognize_ocr('D:/software/label_checker-for-BarTender/test_img/SN.png')
for text in ocr_result:
    print(text)

# 标定图像并裁剪
coordinates = (x1, y1, x2, y2)  # 传入四个坐标
cropped_image = ocr_recognizer.crop_image('image_path', coordinates)
cropped_image.show()