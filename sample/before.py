from PIL import Image
import cv2
from tesserocr import PyTessBaseAPI, RIL

IMAGE_PATH = r'./pics/screen.png'

cv2_image = cv2.imread(IMAGE_PATH)
image = Image.fromarray(cv2_image)

with PyTessBaseAPI(lang='eng+chi_sim') as api:
    api.SetImage(image)
    print(api.AllWordConfidences())
    boxes = api.GetComponentImages(RIL.WORD, True)
    print('Found {} text line image components.'.format(len(boxes)))
    for im, box, *_ in boxes:
        api.SetRectangle(box['x'] - 10, box['y'] - 10, box['w'] + 10, box['h'] + 10)
        ocrResult = api.GetUTF8Text()
        print(f'ocr result: {ocrResult}')
        print(f'box: {box}')
        cv2.rectangle(cv2_image, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (255, 0, 0), 5)

# output
cv2.imwrite('./pics/before.png', cv2_image)
