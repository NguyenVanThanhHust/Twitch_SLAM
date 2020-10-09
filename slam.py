import cv2
import time
import numpy as np
from extractor import Extractor
from frame import Frame, normalize, denormalize, match_frames
import g2o


# camera instrinsics
W = 1280 # 1920 // 2
H = 720 # 1080 // 2

F = 250
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
extractor = Extractor(K)
IRt = np.zeros((3, 4)) # identity trianglation
IRt[:, :3] = np.eye(3)

frames = []

class Point(object):
    def __init__(self, loc):
        self.location = loc
        self.frames = []
    def add_observation(self, frame):
        self.frames.append(frame)

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img):
    img = cv2.resize(img, (960, 540))
    frame = Frame(img, K)
    frames.append(frame)
    if len(frames) <= 1:
        return

    h, w, c = img.shape
    idx1, idx2, pts, Rt = match_frames(frames[-1], frames[-2])
    frames[-1].pose = np.dot(Rt, frames[-2].pose)
    pts4d = triangulate(frames[-1].pose, frames[-2].pose, frames[-1].pts[idx1], frames[-2].pts[idx2])

    # reject pts without enough "parallax"
    good_pts4d = np.abs(pts4d[:, 3]) > 0.005
    print(len(good_pts4d))
    pts4d = pts4d[good_pts4d]
    pts4d /= pts4d[:, 3:]

    # reject pts behind the camera
    good_pts4d = pts4d[:, 2] > 0.005
    pts4d = pts4d[good_pts4d]

    for idx, pt in enumerate(pts4d):
        pt = Point(pt)
        pt.add_observation(frames[-1])
        pt.add_observation(frames[-2])

    for p1, p2 in zip(frames[-1].pts[idx1], frames[-2].pts[idx2]):
        u1, v1 = denormalize(K, p1)
        u2, v2 = denormalize(K, p2)

        cv2.line(img, (u1, v1), (u2, v2), (0, 255, 255), 1)
        cv2.circle(img, (u1,v1), 5, (0, 255, 0), 1)
            
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
