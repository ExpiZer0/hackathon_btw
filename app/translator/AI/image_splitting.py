import cv2
import numpy as np
import os
import pytesseract


image_path = ''  # your image path
input_dir = image_path
output_dir = ''  # your destination path

lower_blue = np.array([30, 30, 30])
upper_blue = np.array([150, 255, 255])

image = cv2.imread(image_path)
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

white_image = np.ones_like(image) * 255

blue_mask_3channel = cv2.merge([blue_mask, blue_mask, blue_mask])

blue_text = cv2.bitwise_and(image, blue_mask_3channel)
white_image[blue_mask_3channel != 0] = blue_text[blue_mask_3channel != 0]

output_path = os.path.join(output_dir, f'processed_{os.path.basename(image_path)}')

processed_image = white_image
# cv2_imshow(processed_image)
rgb = cv2.pyrDown(processed_image)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)
grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

mask = np.zeros(bw.shape, dtype=np.uint8)

roi_list = []

for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.0015 and w > 8 and h > 8:
        roi = rgb[y:y+h, x:x+w]
        roi_list.append(roi)

output_dir = os.path.splitext(output_path)[0]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, roi in enumerate(roi_list):
    roi_path = os.path.join(output_dir, f'roi_{idx+1}.png')
    cv2.imwrite(roi_path, roi)

for roi in roi_list:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2_imshow(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(binary, output_type=Output.DICT, config=custom_config)

    n_boxes = len(details['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
        word_image = roi[y:y+h, x:x+w]
        # Сохранение или дальнейшая обработка каждого слова
        #cv2.imwrite(f'.../processed/example/example_split/word_{i}.png', word_image)

    for i in range(n_boxes):
        (x, y, w, h) = (details['left'][i], details['top'][i], details['width'][i], details['height'][i])
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)