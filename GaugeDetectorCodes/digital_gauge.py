import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract'

img = cv2.imread(r'C:\Users\naveen\Documents\images_for_GaugeDetection\digital_gauge_4.JPG')
img = imutils.resize(img, width=300)
cv2.imshow("original image", img)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed image", gray_img)

gray_img = cv2.bilateralFilter(gray_img, 11, 17, 17)
cv2.imshow("smoothened image", gray_img)

edged = cv2.Canny(gray_img, 30, 200)
cv2.imshow("edged image", edged)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = img.copy()
cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("contours", img1)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
img2 = img.copy()
cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 contours", img2)


i = 7  

for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        break

cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", img)

Cropped_loc = './7.png'
cropped_img = cv2.imread(Cropped_loc)
cv2.imshow("cropped", cropped_img)

plate = pytesseract.image_to_string(cropped_img, lang='eng', config='--psm 7')
plate = plate.strip().replace(" ", "").replace("\n", "").replace("\x0c", "")

if len(plate) >= 5:
    plate = plate[-5:]  
    if plate.endswith(')'):
        plate = plate[:-1]  
else:
    plate = plate.zfill(4)  


plate = plate[:-2] + "." + plate[-2:]

print("Reading:", plate)


cv2.waitKey(0)
cv2.destroyAllWindows()
