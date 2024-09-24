import easyocr

# Create an OCR reader object
reader = easyocr.Reader(['en'])

# Read text from an image
result = reader.readtext('D:/software/label_checker-for-BarTender/test_img/SN.png')

# Print the extracted text
for detection in result:
    print(detection[1])