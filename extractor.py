import cv2
import numpy as np

class Extractor(object):
	GX = 16//2
	GY = 16//2
	def __init__(self):
		self.orb = cv2.ORB_create(100)
		self.bf = cv2.BFMatcher()
		self.last = None

	def extract(self, img):
		#detection
		feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
		
		# extraction
		kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], _size = 20) for f in feats]
		kps, des = self.orb.compute(img, kps)

		# matching
		matches = None
		self.last = {'kps': kps, 'des': des}
		if self.last is not None:
			matches = self.bf.match(des, self.last['des'])

		return kps, des, matches