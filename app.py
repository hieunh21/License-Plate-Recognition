import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
from io import BytesIO

# Import các class từ notebook
from ultralytics import YOLO
import math
import imutils
from skimage import measure
from sklearn.cluster import KMeans
from tensorflow import keras


class ImagePreprocessor:
    """Xử lý tiền xử lý ảnh cơ bản"""

    @staticmethod
    def preprocess(img):
        """Chuyển ảnh sang grayscale và threshold"""
        imgGrayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgGrayscale, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return imgGrayscale, imgThresh

    @staticmethod
    def enhance_with_clahe(img):
        """Tăng cường độ tương phản bằng CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class ImageRotator:
    """Xử lý xoay và cắt ảnh biển số"""

    @staticmethod
    def hough_transform(img, nol=6):
        """Phát hiện đường thẳng bằng Hough Transform"""
        linesP = cv2.HoughLinesP(img, 1, np.pi / 360, 100, None, 100, 20)

        if linesP is not None:
            valid_lines = []
            line_scores = []

            for line in linesP:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue

                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 15 and length > img.shape[1] * 0.3:
                    valid_lines.append(line)
                    line_scores.append(length)

            if valid_lines:
                sorted_indices = np.argsort(line_scores)[::-1]
                valid_lines = [valid_lines[i] for i in sorted_indices[:nol]]
                return np.array(valid_lines)

        return None

    @staticmethod
    def rotation_angle(lines):
        """Tính góc xoay từ các đường thẳng"""
        if lines is None or len(lines) == 0:
            return 0

        angles = []
        weights = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue

            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(angle) < 15:
                angles.append(angle)
                weights.append(length)

        if not angles:
            return 0

        if len(angles) > 3:
            q1, q3 = np.percentile(angles, [25, 75])
            iqr = q3 - q1
            filtered_angles = []
            filtered_weights = []
            for a, w in zip(angles, weights):
                if q1 - 1.5*iqr <= a <= q3 + 1.5*iqr:
                    filtered_angles.append(a)
                    filtered_weights.append(w)
            angles = filtered_angles
            weights = filtered_weights

        if angles and weights:
            return np.average(angles, weights=weights)
        return 0

    @staticmethod
    def rotate_image(img, angle):
        """Xoay ảnh theo góc cho trước"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def find_contours(img):
        """Tìm các contour và sắp xếp theo diện tích"""
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv2.contourArea, reverse=True)

    @staticmethod
    def auto_crop_license_plate(img, binary_img):
        """Tự động cắt vùng biển số"""
        contours = ImageRotator.find_contours(binary_img)
        if not contours:
            return img

        x, y, w, h = cv2.boundingRect(contours[0])
        margin_x, margin_y = int(w*0.05), int(h*0.05)
        H, W = img.shape[:2]
        x1, y1 = max(0, x-margin_x), max(0, y-margin_y)
        x2, y2 = min(W, x+w+margin_x), min(H, y+h+margin_y)
        crop = img[y1:y2, x1:x2]

        return crop

    def auto_rotate_and_crop_lp(self, img):
        """Pipeline đầy đủ: xoay và cắt biển số"""
        if img is None or img.size == 0:
            return None

        gray, th = ImagePreprocessor.preprocess(img)
        canny = cv2.Canny(th, 250, 255)
        kernel = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(canny, kernel, iterations=2)

        lines = self.hough_transform(dil, nol=8)
        if lines is None:
            return img

        angle = self.rotation_angle(lines)
        rot_img = self.rotate_image(img, angle)

        gray_rot = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        th_rot = cv2.threshold(gray_rot, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        final = self.auto_crop_license_plate(rot_img, th_rot)
        return final


class PlateDetector:
    """Phát hiện biển số bằng YOLO"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img, crop_by="bbox"):
        """Phát hiện và cắt biển số từ ảnh"""
        if img is None:
            raise ValueError("Không đọc được ảnh")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)

        bboxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

        masks = getattr(results[0], "masks", None)
        if masks is not None and hasattr(masks, "xy"):
            masks = masks.xy
        else:
            masks = None

        plate_crops = []
        debug_img = img_rgb.copy()

        for i, (bbox, score, class_id) in enumerate(zip(bboxes, scores, class_ids)):
            x1, y1, x2, y2 = map(int, bbox)
            crop = None

            if crop_by == "poly" and masks is not None and i < len(masks):
                polygon = masks[i].astype(np.int32)
                mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)
                crop_masked = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                x, y, w, h = cv2.boundingRect(polygon)
                crop = crop_masked[y:y+h, x:x+w]
            else:
                crop = img_rgb[y1:y2, x1:x2]

            if crop is not None and crop.size > 0:
                plate_crops.append(crop)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Plate {i+1} ({score:.2f})"
                cv2.putText(debug_img, label, (x1, max(y1 - 10, 20)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return plate_crops, debug_img


class PlatePreprocessor:
    """Các phương pháp tiền xử lý ảnh biển số"""

    @staticmethod
    def method1(plate_image):
        """Phương pháp 1 – Threshold Otsu + đảo bit"""
        img_gray_lp = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
        img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))
        img_binary_lp = cv2.bitwise_not(img_binary_lp)
        return img_binary_lp

    @staticmethod
    def method2(plate_image):
        """Phương pháp 2 – Adaptive Threshold (Gaussian) + Morphology"""
        hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        v = cv2.GaussianBlur(v, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel_close = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                cv2.drawContours(thresh, [cnt], -1, 0, -1)

        return thresh

    @staticmethod
    def method3(plate_image):
        """Phương pháp 3 – Otsu + Blur để làm mịn ký tự"""
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        thresh = imutils.resize(thresh, width=400)
        thresh = cv2.medianBlur(thresh, 5)
        thresh = cv2.bitwise_not(thresh)
        return thresh


class CharacterSegmenter:
    """Phát hiện và phân đoạn ký tự"""

    @staticmethod
    def process_character(binary_img, x, y, w, h):
        """Cắt, chuẩn hóa kích thước từng ký tự"""
        char = binary_img[y:y + h, x:x + w]
        char_copy = np.zeros((44, 24))
        char = cv2.resize(char, (20, 40))
        char_copy[2:42, 2:22] = char
        return char_copy

    def find_contours(self, binary_img):
        """Tìm các ký tự có hình dạng hợp lý trong ảnh nhị phân"""
        height, width = binary_img.shape
        char_data = []

        labels = measure.label(binary_img, connectivity=2, background=0)
        regions = measure.regionprops(labels)

        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            h = max_row - min_row
            w = max_col - min_col
            y = min_row
            x = min_col
            aspectRatio = w / float(h) if h > 0 else 0
            area = region.area
            solidity = float(area) / (w * h) if w * h > 0 else 0
            heightRatio = h / float(height)

            if (w >= 8 and h >= 15 and w < width / 1.8 and h < height / 1.1 and
                    0.15 < aspectRatio < 1.2 and solidity > 0.3 and
                    0.15 < heightRatio < 0.95):
                char_data.append({
                    'image': self.process_character(binary_img, x, y, w, h),
                    'x': x, 'y': y, 'w': w, 'h': h, 'center_y': y + h / 2
                })

        if len(char_data) < 6:
            inverted_binary = cv2.bitwise_not(binary_img.copy())
            processed = cv2.medianBlur(inverted_binary, 3)
            contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspectRatio = w / float(h) if h > 0 else 0
                contour_area = cv2.contourArea(contour)
                solidity = contour_area / float(w * h) if w * h > 0 else 0
                heightRatio = h / float(height)

                if (w >= 8 and h >= 15 and w < width / 1.8 and h < height / 1.1 and
                        0.15 < aspectRatio < 1.2 and solidity > 0.25 and
                        0.15 < heightRatio < 0.95):
                    char_data.append({
                        'image': self.process_character(binary_img, x, y, w, h),
                        'x': x, 'y': y, 'w': w, 'h': h, 'center_y': y + h / 2
                    })

        return char_data

    @staticmethod
    def determine_plate_type_and_order(char_data, plate_height):
        """Xác định kiểu biển (1 hàng hay 2 hàng) và sắp xếp thứ tự ký tự"""
        if not char_data:
            return [], 'unknown'

        y_centers = np.array([char['center_y'] for char in char_data])

        if len(y_centers) > 3:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(
                y_centers.reshape(-1, 1)
            )
            centers = kmeans.cluster_centers_.flatten()
            labels = kmeans.labels_
            vertical_distance = abs(centers[0] - centers[1])

            if vertical_distance > plate_height * 0.25:
                upper_chars = [char for char, label in zip(char_data, labels)
                              if label == (0 if centers[0] < centers[1] else 1)]
                lower_chars = [char for char, label in zip(char_data, labels)
                              if label == (1 if centers[0] < centers[1] else 0)]
                upper_chars.sort(key=lambda c: c['x'])
                lower_chars.sort(key=lambda c: c['x'])
                sorted_chars = upper_chars + lower_chars
                return [char['image'] for char in sorted_chars], 'two-line'

        char_data.sort(key=lambda c: c['x'])
        return [char['image'] for char in char_data], 'one-line'


class QualityEvaluator:
    """Đánh giá chất lượng preprocessing"""

    @staticmethod
    def calculate_quality_score(result, model, char_dict):
        """Tính điểm chất lượng cho mỗi phương pháp preprocessing"""
        score = 0
        char_count = result["char_count"]
        char_data = result["char_data"]

        if 7 <= char_count <= 9:
            score += 40
        elif 6 <= char_count <= 10:
            score += 30
        elif char_count == 5 or char_count == 11:
            score += 20
        else:
            score += max(0, 10 - abs(char_count - 7.5) * 2)

        if char_data:
            confidences = []
            for char_info in char_data:
                char_img = char_info['image']
                img = cv2.resize(char_img, (28, 28)).reshape(1, 28, 28, 1)
                pred = model.predict(img, verbose=0)
                max_conf = np.max(pred)
                confidences.append(max_conf)

            avg_confidence = np.mean(confidences) if confidences else 0
            score += avg_confidence * 30

        if char_data and len(char_data) > 1:
            heights = [c['h'] for c in char_data]
            widths = [c['w'] for c in char_data]

            height_std = np.std(heights)
            width_std = np.std(widths)

            uniformity_score = max(0, 20 - min(height_std + width_std, 20))
            score += uniformity_score

        if char_data and len(char_data) >= 2:
            x_positions = sorted([c['x'] for c in char_data])
            gaps = [x_positions[i+1] - x_positions[i]
                   for i in range(len(x_positions)-1)]
            if gaps:
                gap_std = np.std(gaps)
                spacing_score = max(0, 10 - min(gap_std / 2, 10))
                score += spacing_score

        return score

    def select_best_preprocessing(self, all_results, model, char_dict):
        """Chọn phương pháp preprocessing tốt nhất"""
        if not all_results:
            return None

        for result in all_results:
            result["quality_score"] = self.calculate_quality_score(
                result, model, char_dict
            )

        all_results.sort(key=lambda r: r["quality_score"], reverse=True)
        best_result = all_results[0]
        return best_result


class LicensePlateRecognizer:
    """Class chính để nhận dạng biển số xe"""

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        self.char_dict = {i: c for i, c in enumerate(
            list("ABCDEFGHKLMNPRSTUVXYZ0123456789") + ["Background"]
        )}
        self.rotator = ImageRotator()
        self.segmenter = CharacterSegmenter()
        self.evaluator = QualityEvaluator()

    def recognize(self, plate_image):
        """Pipeline đầy đủ: nhận dạng biển số từ ảnh đã crop"""
        plate_height, plate_width = plate_image.shape[:2]
        aspect_ratio = plate_width / plate_height

        plate_image = ImagePreprocessor.enhance_with_clahe(plate_image)

        if aspect_ratio > 2.0:
            plate_image = cv2.resize(plate_image, (350, 100))
        else:
            plate_image = cv2.resize(plate_image, (240, 200))

        rotated_lp = self.rotator.auto_rotate_and_crop_lp(plate_image)
        if rotated_lp is None:
            return "", None, []

        preprocessing_methods = [
            {"name": "Method 1", "function": PlatePreprocessor.method1},
            {"name": "Method 2", "function": PlatePreprocessor.method2},
            {"name": "Method 3", "function": PlatePreprocessor.method3},
        ]

        all_results = []
        for method in preprocessing_methods:
            img_binary = method["function"](rotated_lp)
            char_data = self.segmenter.find_contours(img_binary)
            all_results.append({
                "method_name": method["name"],
                "char_count": len(char_data),
                "char_data": char_data,
                "img_binary": img_binary
            })

        best_result = self.evaluator.select_best_preprocessing(
            all_results, self.model, self.char_dict
        )
        best_char_data = best_result["char_data"]

        segmented_chars, plate_type = self.segmenter.determine_plate_type_and_order(
            best_char_data, best_result["img_binary"].shape[0]
        )

        output = []
        confidences = []

        for idx, char_img in enumerate(segmented_chars):
            img = cv2.resize(char_img, (28, 28)).reshape(1, 28, 28, 1)
            pred = self.model.predict(img, verbose=0)
            ci = np.argmax(pred)
            confidence = pred[0][ci]

            if ci != 31 and confidence > 0.5:
                output.append(self.char_dict[ci])
                confidences.append(confidence)

        plate_number = ''.join(output)
        avg_conf = np.mean(confidences) if confidences else 0

        return plate_number, avg_conf, segmented_chars


class LicensePlateSystem:
    """Class tổng hợp toàn bộ hệ thống nhận dạng biển số"""

    def __init__(self, yolo_model_path, ocr_model_path):
        self.detector = PlateDetector(yolo_model_path)
        self.recognizer = LicensePlateRecognizer(ocr_model_path)

    def process_image(self, img, crop_by="bbox"):
        """Xử lý ảnh: detect -> nhận dạng"""
        plate_crops, debug_img = self.detector.detect(img, crop_by)

        results = []
        for i, plate_img in enumerate(plate_crops):
            plate_number, confidence, chars = self.recognizer.recognize(plate_img)
            results.append({
                'plate_number': plate_number,
                'confidence': confidence,
                'plate_image': plate_img,
                'characters': chars
            })

        return results, debug_img


# ============================================================
# STREAMLIT WEB APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Nhận diện biển số xe",
        page_icon="🚗",
        layout="wide"
    )

    st.title("🚗 Hệ thống nhận diện biển số xe")
    st.markdown("---")

    # Sidebar - Cấu hình
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        
        yolo_model_path = st.text_input(
            "Đường dẫn YOLO model",
            value="model/best.pt"
        )
        
        ocr_model_path = st.text_input(
            "Đường dẫn OCR model",
            value="model/weight.keras"
        )
        
        crop_method = st.selectbox(
            "Phương pháp cắt biển số",
            ["bbox", "poly"],
            help="bbox: Bounding Box, poly: Polygon"
        )
        
        st.markdown("---")
        st.markdown("### 📝 Hướng dẫn")
        st.markdown("""
        1. Tải lên ảnh chứa biển số xe
        2. Nhấn nút **Nhận diện**
        3. Xem kết quả và độ tin cậy
        """)

    # Khởi tạo session state
    if 'lp_system' not in st.session_state:
        st.session_state.lp_system = None
    
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

    # Load models
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if not st.session_state.models_loaded:
            if st.button("🔧 Tải Models", type="primary"):
                with st.spinner("Đang tải models..."):
                    try:
                        if not os.path.exists(yolo_model_path):
                            st.error(f"❌ Không tìm thấy YOLO model: {yolo_model_path}")
                            return
                        if not os.path.exists(ocr_model_path):
                            st.error(f"❌ Không tìm thấy OCR model: {ocr_model_path}")
                            return
                        
                        st.session_state.lp_system = LicensePlateSystem(
                            yolo_model_path, ocr_model_path
                        )
                        st.session_state.models_loaded = True
                        st.success("✅ Models đã được tải thành công!")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi tải models: {str(e)}")
        else:
            st.success("✅ Models đã sẵn sàng")
            if st.button("🔄 Tải lại Models"):
                st.session_state.models_loaded = False
                st.session_state.lp_system = None
                st.rerun()

    st.markdown("---")

    # Upload ảnh
    uploaded_file = st.file_uploader(
        "📤 Chọn ảnh chứa biển số xe",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff'],
        help="Hỗ trợ định dạng: JPG, JPEG, PNG, WEBP, BMP, TIFF"
    )

    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        # Chuyển đổi sang RGB nếu là RGBA (cho WEBP và PNG có alpha)
        if image.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            background.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Ảnh gốc")
            st.image(image, use_container_width=True)
        
        # Nút nhận diện
        if st.session_state.models_loaded:
            if st.button("🔍 Nhận diện biển số", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    try:
                        # Xử lý
                        results, debug_img = st.session_state.lp_system.process_image(
                            img_bgr, crop_by=crop_method
                        )
                        
                        # Hiển thị ảnh detection
                        with col2:
                            st.subheader("🎯 Phát hiện biển số")
                            # debug_img đã là RGB từ PlateDetector.detect()
                            st.image(debug_img, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Hiển thị kết quả
                        if results:
                            st.subheader(f"✨ Kết quả nhận diện ({len(results)} biển số)")
                            
                            for idx, result in enumerate(results):
                                with st.expander(f"🚗 Biển số #{idx + 1}: **{result['plate_number']}**", expanded=True):
                                    col1, col2, col3 = st.columns([2, 2, 3])
                                    
                                    with col1:
                                        st.markdown("**Biển số phát hiện:**")
                                        # plate_image đã là RGB từ PlateDetector.detect()
                                        st.image(result['plate_image'], use_container_width=True)
                                    
                                    with col2:
                                        st.markdown("**Thông tin:**")
                                        st.metric("Biển số", result['plate_number'])
                                        if result['confidence']:
                                            confidence_pct = result['confidence'] * 100
                                            st.metric("Độ tin cậy", f"{confidence_pct:.1f}%")
                                            
                                            # Progress bar cho confidence
                                            st.progress(float(result['confidence']))
                                    
                                    with col3:
                                        if result['characters']:
                                            st.markdown("**Ký tự phát hiện:**")
                                            # Hiển thị các ký tự
                                            n_chars = len(result['characters'])
                                            cols = st.columns(min(n_chars, 10))
                                            for i, char_img in enumerate(result['characters'][:10]):
                                                with cols[i]:
                                                    st.image(char_img, use_container_width=True, clamp=True)
                        else:
                            st.warning("⚠️ Không phát hiện được biển số nào trong ảnh")
                            
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý: {str(e)}")
                        st.exception(e)
        else:
            st.warning("⚠️ Vui lòng tải models trước khi nhận diện")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Hệ thống nhận diện biển số xe - Sử dụng YOLO + CNN</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
