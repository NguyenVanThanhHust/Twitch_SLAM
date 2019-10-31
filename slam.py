import cv2
import time
import numpy as np
from extractor import Extractor

W = 960
H = 540

fe = Extractor()
def process_frame(img):
	img = cv2.resize(img, (960, 540))
	# find the keypoints with ORB
	kps, des, matches = fe.extract(img)
	for p in kps:
		u,v = map(lambda x: int(round(x)), p.pt)
		cv2.circle(img, (u,v), 5, (0, 255, 0), 1)
			
	# for point in kp:
	# 	u,v = map(lambda x: int(round(x)), point.pt)
	# 	cv2.circle(img, (u,v), 5, (0, 255, 0), 1)

	cv2.imshow("result", img)
	cv2.waitKey(1)

if __name__ == '__main__':
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret , frame = cap.read()
		if ret == True:
			process_frame(frame)
			# Press Q on keyboard to  exit
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break