from findtext import FindText
import cv2

SAMPLE_IMG_NAME = './pics/screen.png'

ft = FindText(lang='chi_sim')

r = ft.find_word(SAMPLE_IMG_NAME, deep=True, offset=5)
cv2_object = cv2.imread(SAMPLE_IMG_NAME)
for each in r:
    print(each.location)
    cv2.rectangle(cv2_object, *each.location, (255, 0, 0), 5)

cv2.imwrite('./pics/after.png', cv2_object)
