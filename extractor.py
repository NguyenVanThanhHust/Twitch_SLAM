import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
np.set_printoptions(suppress=True)

def add_one(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    # GX = 16//2
    # GY = 12//2
    def __init__(self, K):

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_one(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))

        # print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        #detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
        
        # extraction
        kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], _size = 20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        print(kps)
        # matching
        matches = None
        self.last = {'kps': kps, 'des': des}
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k = 2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    print(kp1, kp2)
                    ret.append((kp1, kp2))

        
        # filter
        if len(ret) > 0:
            ret = np.asarray(ret)
            # normalize coordes, subtract to move to 
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliners = ransac((ret[:, 0], ret[:, 1]), 
                                        # FundamentalMatrixTransform, 
                                        EssentialMatrixTransform, 
                                        min_samples = 8, 
                                        residual_threshold = 0.01, 
                                        max_trials = 100)
            ret = ret[inliners]

            s, v, d = np.linalg.svd(model.params)
            f_est = np.sqrt(2)/((v[0] + v[1])/2)

        self.last = {'kps': kps, 'des': des}
        return ret
