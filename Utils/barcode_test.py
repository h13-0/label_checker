from pyzbar.pyzbar import decode
from PIL import Image
# 通过条码barcode获取SN码

# 获取解码的结果
result = decode(Image.open('D:/software/label_checker-for-BarTender/test_img/barcode.png'))

# 提取数据部分并打印
if result:
    barcode_data = result[0].data.decode('utf-8')  # 解码字节数据并转换为字符串
    print("Barcode data:", barcode_data)
else:
    print("No barcode detected.")