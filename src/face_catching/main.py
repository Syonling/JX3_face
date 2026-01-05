import cv2 as cv
from face_catching.tracker import FaceTracker
from face_catching.current_features import FeatureExtractor
from face_catching.visualizer import Visualizer

def main():
    tracker = FaceTracker(cam_index=1)  #做人脸检测
    extractor = FeatureExtractor()      #算特征
    vis = Visualizer()                  #画点/文字

    indices = [33, 133, 159, 145, 61, 291, 13, 14, 105]

    while True:
        ret, frame = tracker.read()     #frame 每一帧的 BGR 图像（OpenCV 格式）
        if not ret:
            break

        face_lms = tracker.process(frame)   #这一帧检测到的人脸 landmarks（468+点）

        if face_lms:
            cv.putText(frame, "FACE", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            vis.draw_points(frame, face_lms, indices)

            feats = extractor.extract(face_lms, frame.shape[1], frame.shape[0]) #从 face_lms 算出 gaze 等特征
            vis.draw_gaze(frame, feats["gaze_x"], feats["gaze_y"], feats["iris_left"], feats["iris_right"])
        else:
            cv.putText(frame, "NO FACE", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv.imshow("frame", frame)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    tracker.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()