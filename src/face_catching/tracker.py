import cv2 as cv
import mediapipe as mp

class FaceTracker:
    """
    •	负责“输入端”：摄像头与 MediaPipe 人脸跟踪
	•	输出：face_lms（landmarks）或 None
    """
    def __init__(self, cam_index=1):
        self.cap = cv.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        mp_face_mesh = mp.solutions.face_mesh   #初始化 MediaPipe FaceMesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def read(self):
        """
        作用：
	    •	从摄像头读取一帧
        返回：
            •	ret：是否成功
            •	frame：BGR 图像帧
        """
        return self.cap.read()

    def process(self, frame_bgr):
        """
        	•	把 BGR 转成 RGB（MediaPipe 需要 RGB）
            •	跑 face_mesh.process
        返回：
            •	如果检测到人脸：返回 face_lms
            •	没检测到：返回 None
        """
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

    def release(self):
        self.cap.release()