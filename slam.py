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
	matches = fe.extract(img)
	for p1, p2 in matches:
		u1, v1 = map(lambda x: int(round(x)), p1)
		u2, v2 = map(lambda x: int(round(x)), p2)
		cv2.line(img, (u1, v1), (u2, v2), (0, 255, 0), 1)
		# cv2.circle(img, (u,v), 5, (0, 255, 0), 1)
			
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
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break