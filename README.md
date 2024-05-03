# Study_ocr
### 사용 서비스

google colab(easyocr), vscode(tesseract)

### 설명 

vscode에서 tesseract를 활용하여 이미지를 불러와 변환을 주고 텍스츠로 추출.

변환 시킨 이미지를 스캔 후 colab에서 실행하였다.

⬇ tesseract ocr 코드
``` python
import pytesseract
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
result = open(r"C:/img/output.txt", "w", encoding='utf-8')

path_dir = r'<이미지 경로>'
file_list = os.listdir(path_dir)


for file_name in file_list :
    if file_name == "output.txt":
        continue

    img_path = os.path.join(path_dir, file_name)
    img = cv2.imread(img_path)

    val = 155
    array = np.full(img.shape, (val,val,val), dtype=np.uint8)

    bright_img = cv2.add(img, array)

    gray = cv2.cvtColor(bright_img, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    save_path = '<이미지 저장 경로>'
    cv2.imwrite(save_path, binary)
 

    extracted_text = pytesseract.image_to_string(save_path, lang='KOR+ENG', 
                                                  config=r'-c preserve_interword_spaces=1 --psm 3 --oem 3 -l kor+eng --tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"')
    
    result.write(extracted_text + '\n')


result.close()
print("추출이 완료되었습니다. 확인부탁드립니다.")
```

⬇ image_scan 코드
``` python
import cv2
import os
import numpy as np

img_path = '<이미지 경로>'
filename, ext = os.path.splitext(os.path.basename(img_path))
ori_img = cv2.imread(img_path)

src = []

# mouse callback handler
def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        img = ori_img.copy()

        src.append([x, y])

        for xx, yy in src:
            cv2.circle(img, center=(xx, yy), radius=5, color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow('img', img)

        # perspective transform
        if len(src) == 4:
            src_np = np.array(src, dtype=np.float32)

            width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
            height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

            dst_np = np.array([
                [0, 0],
                [width, 0],
                [width, height],
                [0, height]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
            result = cv2.warpPerspective(ori_img, M=M, dsize=(int(width), int(height)))

            cv2.imshow('result', result)
            cv2.imwrite('<이미지 저장 경로>%s<저장 할 이름>%s' % (filename, ext), result)

cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_handler)

cv2.imshow('img', ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 원본 이미지

<img src="https://github.com/jjang-yu/Study_ocr/assets/160578079/155db0e8-dda4-4511-b31c-a5b24e769c72" width="400" height="250"/>

#### 변환 이미지

<img src="https://github.com/jjang-yu/Study_ocr/assets/160578079/08332f46-0b2b-4843-b45d-794949a2f267" width="400" height="250"/>

#### 스캔 이미지

<img src="https://github.com/jjang-yu/Study_ocr/assets/160578079/6b249ecf-d98c-4625-9282-46bd2ac8eb9d" width="200" height="300"/>




