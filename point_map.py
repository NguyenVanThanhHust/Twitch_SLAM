import sys
import numpy as np
import OpenGL.GL as gl
import pangolin
import g2o
from multiprocessing import Process, Queue
from frame import poseRt

class Point(object):
  # A Point is a 3-D point in the world
  # Each Point is observed in multiple Frames

    def __init__(self, mapp, loc, color):
        self.pt = loc
        self.frames = []
        self.idxs = []
        self.color = np.copy(color)
        self.id = len(mapp.points)
        mapp.points.append(self)

    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)

class Map(object):
    def __init__(self):
        self.frames = []
        self.points = []
        self.state = None
        self.q = Queue()

    def optimize(self):
        # init g2o solver
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # add frames to graph
        for frame in self.frames:
            pose = frame.pose
            sbacam = g2o.SBACam(g2o.SE3Quat(frame.pose[0:3, 0:3], frame.pose[0:3, 3]))
            # sbacam.set_cam(frame.K[0][0], frame.K[1][1], frame.K[2][0], frame.K[2][1], 1.0)
            sbacam.set_cam(1.0, 1.0, 0.0, 0.0, 1.0)
            v_se3 = g2o.VertexCam()
            v_se3.set_id(frame.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(frame.id == 0)
            opt.add_vertex(v_se3)

        # add points to frames
        PT_ID_OFFSET = 0x10000
        for point in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(point.id + PT_ID_OFFSET)
            pt.set_estimate(point.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)
            
            for frame in point.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(frame.id))
                uv = frame.kps[frame.pts.index(point)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)
        # opt.set_verbose(True)                
        opt.initialize_optimization()
        opt.optimize(20)

        # add pose back to frame
        for frame in self.frames:
            est = opt.vertex(frame.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            frame.pose = poseRt(R, t)

    def create_viewer(self):
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()
        
    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        pangolin.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                  0, 0, 0,
                                  0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
        self.dcam.SetHandler(self.handler)

    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        # gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.dcam.Activate(self.scam)

        # draw poses
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])

        # draw keypoints
        gl.glPointSize(5)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1], self.state[2])

        pangolin.FinishFrame()

    def display(self):
        poses, pts, colors = [], [], []
        for f in self.frames:
            poses.append(f.pose)
        for p in self.points:
            pts.append(p.pt)
            colors.append(p.color)
        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))


