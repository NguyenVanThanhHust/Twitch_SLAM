import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform
np.set_printoptions(suppress=True)

# def add_one(x):
    # return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
def add_one(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        one_matrix = np.ones((x.shape[0], 1))
        return np.concatenate([x, one_matrix], axis=1)

def extractRt(EssentialMatrix):
    E = EssentialMatrix
    U, w, Vt = np.linalg.svd(E)

    # assert np.linalg.det(U) > 0, "check det of U"
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return Rt

def normalize(Kinv, pts):
    pts_add1 = add_one(pts)
    print(pts_add1.shape)
    return np.dot(Kinv, pts_add1.T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def extract(img):
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
    orb = cv2.ORB_create()
    # extraction
    kps = [cv2.KeyPoint(x = f[0][0], y = f[0][1], _size = 20) for f in pts]
    kps, des = orb.compute(img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):
    #detection
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k = 2)
    
    ret = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))

    # filter
    assert len(ret) >= 8
    ret = np.asarray(ret)
    # normalize coordes, subtract to move to 
    # ret[:, 0, :] = normalize(Kinv, ret[:, 0, :])
    # ret[:, 1, :] = normalize(Kinv, ret[:, 1, :])

    model, inliners = ransac((ret[:, 0], ret[:, 1]), 
                                # FundamentalMatrixTransform, 
                                EssentialMatrixTransform, 
                                min_samples = 8, 
                                residual_threshold = 0.01, 
                                max_trials = 100)
    ret = ret[inliners]
    Rt = extractRt(model.params)
    return ret, Rt

class Frame(object):
    def __init__(self, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, pts)

class Extractor(object):
    # GX = 16//2
    # GY = 12//2
    def __init__(self, K):

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        # print(ret)
        return int(round(ret[0])), int(round(ret[1]))

