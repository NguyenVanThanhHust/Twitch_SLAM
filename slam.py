import cv2
import time
import numpy as np
from extractor import Extractor

W = 1280 # 1920 // 2
H = 720 # 1080 // 2

F = 250
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img, (960, 540))
    h, w, c = img.shape
    # find the keypoints with ORB
    matches = fe.extract(img)
    for p1, p2 in matches:
        u1, v1 = fe.denormalize(p1)
        u2, v2 = fe.denormalize(p2)

        cv2.line(img, (u1, v1), (u2, v2), (0, 255, 255), 1)
        cv2.circle(img, (u1,v1), 5, (0, 255, 0), 1)
            
    # filter

    cv2.imshow("result", img)
    cv2.waitKey(1)

if __name__ == '__main__':
    cap = cv2.VideoCapture("test.mp4")
    while cap.isOpened():
        ret , frame = cap.read()
        if ret == True:
            process_frame(frame)
            # Press Q on keyboard to  exit
            k = cv2.waitKey(1)
            if k == 27:
                break
        else:
            break
