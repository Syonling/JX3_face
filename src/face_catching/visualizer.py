import cv2 as cv
from face_catching.current_features import lm_xy

class Visualizer:
    def draw_points(self, frame, face_lms, indices):
        """
        	•	把 indices 这一组点画在画面上
        输入：
            •	frame：要画的图
            •	face_lms：landmarks
            •	indices：要画的点编号列表
        """
        h, w = frame.shape[:2]
        for idx in indices:
            x, y = lm_xy(face_lms, idx, w, h)
            cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

    def draw_gaze(self, frame, gaze_x, gaze_y, iris_left, iris_right):
        """
        •	画虹膜中心点（紫色）
	    •	画 gaze 数值文字（gaze_x / gaze_y）

        """
        cv.circle(frame, iris_left, 3, (255, 0, 255), -1)
        cv.circle(frame, iris_right, 3, (255, 0, 255), -1)
        cv.putText(frame, f"gaze_x: {gaze_x:+.3f}", (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"gaze_y: {gaze_y:+.3f}", (10, 85),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)