import cv2

def process_frame(img):
	img = cv2.resize(img, (540, 960))
	print(img.shape)

if __name__ == '__main__':
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret , frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break