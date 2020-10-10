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
    print(Rt.shape)
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def normalize(Kinv, pts):
    pts_add1 = add_one(pts)
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
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))

    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    # filter
    assert len(ret) >= 8
    ret = np.asarray(ret)
    model, inliners = ransac((ret[:, 0], ret[:, 1]), 
                                # FundamentalMatrixTransform, 
                                EssentialMatrixTransform, 
                                min_samples = 8, 
                                residual_threshold = 0.01, 
                                max_trials = 100)
    # ret = ret[inliners]
    Rt = extractRt(model.params)
    idx1 = idx1[inliners]
    idx2 = idx2[inliners]

    return idx1, idx2, ret, Rt

class Frame(object):
    def __init__(self, mapp, img, K, pose=np.eye(4)):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        pts, self.des = extract(img)
        self.pts = normalize(self.Kinv, pts)
        self.pose = pose
        self.id = len(mapp.frames)
        mapp.frames.append(self)
        
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

