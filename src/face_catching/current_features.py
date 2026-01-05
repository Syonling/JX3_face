def lm_xy(face_lms, idx, w, h) -> int | int:
    """
    把 MediaPipe 的归一化 landmark（0~1）转换成像素坐标
    输入：
        •	face_lms：当前帧 landmarks
        •	idx：点编号
        •	w, h：图像宽高
    输出：
        •	(x, y)：像素坐标（int）
    """
    lm = face_lms.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

class FeatureExtractor:
    """
    从 landmarks 计算当前特征
    """
    def extract(self, face_lms, w, h) -> dict:
        """
        把这一帧的特征打包成一个 dict
        """
        (gaze_x, gaze_y), (iris_left, iris_right) = self.compute_gaze(face_lms, w, h)
        return {
            "gaze_x": gaze_x,       #归一化后的注视方向（近似）
            "gaze_y": gaze_y,
            "iris_left": iris_left,     # (x, y) 虹膜中心坐标（用于画出来验证）
            "iris_right": iris_right,   # (x, y)
        }

    def compute_gaze(self, face_lms, w, h):
        iris_a = lm_xy(face_lms, 468, w, h)
        iris_b = lm_xy(face_lms, 473, w, h)
        iris_left, iris_right = (iris_a, iris_b) if iris_a[0] < iris_b[0] else (iris_b, iris_a)

        re_inner = lm_xy(face_lms, 133, w, h)
        re_outer = lm_xy(face_lms, 33,  w, h)
        re_top   = lm_xy(face_lms, 159, w, h)
        re_bot   = lm_xy(face_lms, 145, w, h)

        le_inner = lm_xy(face_lms, 362, w, h)
        le_outer = lm_xy(face_lms, 263, w, h)
        le_top   = lm_xy(face_lms, 386, w, h)
        le_bot   = lm_xy(face_lms, 374, w, h)

        def norm_pos(iris, inner, outer, top, bot):
            """
            •	把虹膜中心在“眼框坐标系”里归一化
            •	输出 (gx, gy)：
            •	gx：左右偏移（相对眼角中心，除以眼宽）
            •	gy：上下偏移（相对上下眼睑中心，除以眼高）
            """
            dx = (outer[0] - inner[0]) if abs(outer[0] - inner[0]) > 1 else 1
            dy = (bot[1] - top[1])     if abs(bot[1] - top[1]) > 1 else 1
            cx = (inner[0] + outer[0]) / 2.0
            cy = (top[1] + bot[1])     / 2.0
            gx = (iris[0] - cx) / dx
            gy = (iris[1] - cy) / dy
            return float(gx), float(gy)

        gaze_rx, gaze_ry = norm_pos(iris_right, re_inner, re_outer, re_top, re_bot)
        gaze_lx, gaze_ly = norm_pos(iris_left,  le_inner, le_outer, le_top, le_bot)

        gaze_x = (gaze_rx + gaze_lx) / 2.0
        gaze_y = (gaze_ry + gaze_ly) / 2.0
        return (gaze_x, gaze_y), (iris_left, iris_right)