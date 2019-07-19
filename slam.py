import cv2
import pygame
import time

W = 960
H = 540
pygame.init()
display = pygame.display.set_mode((W,H))

def process_frame(img):
	img = cv2.resize(img, (960, 540))
	surf = pygame.surfarray.make_surface(img.swapaxes(0,1)).convert()

	display.blit(surf, (0, 0))
	pygame.display.update()
	#time.sleep(1)


if __name__ == '__main__':
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret , frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break