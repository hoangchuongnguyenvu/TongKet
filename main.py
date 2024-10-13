import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from streamlit_drawable_canvas import st_canvas

# Ứng dụng Phân đoạn ảnh với Watershed
def watershed_app():
    st.title("Ứng dụng Phân đoạn ảnh với Watershed")

    st.header("1. Dataset")
    # Tải sẵn 4 ảnh
    images = [
        cv2.imread('images/ndata616.jpg'),
        cv2.imread('images/ndata20.jpg'),
        cv2.imread('images/ndata617.jpg'),
        cv2.imread('images/ndata29.jpg')
    ]

    col1, col2, col3, col4 = st.columns(4)
    for i, image in enumerate(images, 1):
        with locals()[f'col{i}']:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f'Ảnh {i}', use_column_width=True)

    st.header("2. Giới thiệu phương pháp Watershed")
    st.write("""
    Watershed là một thuật toán phân đoạn ảnh dựa trên nguyên lý địa hình học. Nó coi ảnh như một bề mặt địa hình, 
    trong đó cường độ pixel đại diện cho độ cao. Thuật toán này tìm kiếm các "lưu vực" (basins) và "đường phân thủy" (watersheds) 
    trên bề mặt này để phân đoạn ảnh thành các vùng khác nhau.

    Các bước chính của thuật toán Watershed:
    1. Tiền xử lý ảnh (làm mịn, loại bỏ nhiễu).
    2. Tính toán gradient của ảnh.
    3. Đánh dấu các vùng chắc chắn là foreground và background.
    4. Áp dụng biến đổi khoảng cách.
    5. Thực hiện quá trình "ngập nước" từ các điểm đánh dấu.
    6. Xác định các đường phân thủy để phân đoạn ảnh.

    Watershed rất hiệu quả trong việc phân đoạn các đối tượng chạm nhau trong ảnh.
    """)

    st.header("3. Kiểm thử")
    st.subheader("Giới thiệu IoU (Intersection over Union)")
    st.write("""
    IoU (Intersection over Union) là một metric phổ biến để đánh giá chất lượng của các thuật toán phân đoạn ảnh. 
    IoU đo lường mức độ chồng lấp giữa vùng được dự đoán và vùng ground truth.

    IoU được tính bằng công thức:
    IoU = (Vùng giao) / (Vùng hợp)

    Giá trị IoU nằm trong khoảng từ 0 đến 1, trong đó:
    - IoU = 1 nghĩa là dự đoán hoàn hảo
    - IoU = 0 nghĩa là không có sự chồng lấp nào giữa dự đoán và ground truth

    Thông thường, IoU > 0.5 được coi là một dự đoán tốt.
    """)

    # Tải sẵn ảnh ground truth
    ground_truth = cv2.imread('IoU.png')
    st.image(ground_truth, caption='Minh họa bằng hình ảnh', use_column_width=True)

    st.header("4. Tham số huấn luyện")
    st.write("""
    Các tham số huấn luyện không thay đổi trong thuật toán Watershed bao gồm:
    - Kích thước kernel cho morphological operations: (3,3)
    - Số lần lặp cho opening: 2
    - Số lần lặp cho dilate: 3
    - Phương pháp tính khoảng cách: cv2.DIST_L2
    - Hệ số ngưỡng cho sure foreground: 0.7
             
    Các tham số huấn luyện thay đổi trong quá trình huấn luyện với thuật toán Watershed bao gồm:
    - Threshold :(0,255,5)
    - Min Distance : (1,51,2) 
    """)

    st.header("5. Sự thay đổi của phương pháp đánh giá IoU theo từng tham số huán luyện")
    # Tải sẵn 2 ảnh đầu vào
    input_image1 = cv2.imread('Metirc/IvMD.png')
    input_image2 = cv2.imread('Metirc/IvThres.png')

    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(input_image1, cv2.COLOR_BGR2RGB), caption='Min Distance', use_column_width=True)
    with col2:
        st.image(cv2.cvtColor(input_image2, cv2.COLOR_BGR2RGB), caption='Threshold', use_column_width=True)

    st.write("""
    Kết quả huấn luyện với bộ tham số tốt nhất
    """)
    input_image3 = cv2.imread('Metirc/MinDistance.png')
    input_image4 = cv2.imread('Metirc/Threshold.png')

    col3, col4 = st.columns(2)
    with col3:
        st.image(cv2.cvtColor(input_image3, cv2.COLOR_BGR2RGB), caption='Min Distance tốt nhất', use_column_width=True)
    with col4:
        st.image(cv2.cvtColor(input_image4, cv2.COLOR_BGR2RGB), caption='Threshold tốt nhất', use_column_width=True)

    st.header("6. Ứng dụng phân đoạn ảnh với Watershed")

    # Cho phép người dùng tải lên 1 ảnh
    uploaded_image = st.file_uploader("Tải lên ảnh để phân đoạn", type=["jpg", "jpeg", "png"])

    if st.button("Chạy phân đoạn ảnh"):
        if uploaded_image is not None:
            # Đọc ảnh từ file đã tải lên
            image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

            # Chuyển đổi ảnh sang grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Áp dụng threshold
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

            # Sure background area
            sure_bg = cv2.dilate(opening, kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now, mark the region of unknown with zero
            markers[unknown == 255] = 0

            # Apply watershed
            markers = cv2.watershed(image, markers)

            # Create a color image to draw on
            segmented = np.zeros(image.shape, dtype=np.uint8)

            # Fill segmented areas with random colors
            for label in range(2, np.max(markers) + 1):
                segmented[markers == label] = np.random.randint(0, 255, 3)

            # Mark watershed boundaries
            segmented[markers == -1] = [255, 255, 255]

            # Hiển thị kết quả
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title('Ảnh gốc')
            ax1.axis('off')
            
            ax2.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
            ax2.set_title('Kết quả phân đoạn')
            ax2.axis('off')

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("Vui lòng tải lên một ảnh trước khi chạy phân đoạn.")

# Ứng dụng Phát hiện Khuôn mặt
@st.cache_resource
def load_detector():
    return MTCNN()

def face_detection_app():
    st.title("Ứng dụng Phát hiện Khuôn mặt")

    detector = load_detector()

    def detect_faces(image):
        # Phát hiện khuôn mặt bằng MTCNN
        faces = detector.detect_faces(image)
        return faces

    uploaded_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Đọc ảnh từ file đã tải lên
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Phát hiện khuôn mặt
        faces = detect_faces(image_np)

        if len(faces) == 0:
            st.write("Không tìm thấy khuôn mặt nào trong ảnh.")
        else:
            # Vẽ hình chữ nhật xung quanh khuôn mặt
            result_image = image_np.copy()
            for face in faces:
                x, y, width, height = face['box']
                cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Hiển thị số lượng khuôn mặt
            cv2.putText(result_image, f'Faces detected: {len(faces)}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Hiển thị ảnh kết quả
            st.image(result_image, caption='Ảnh với khuôn mặt được phát hiện', use_column_width=True)

            st.write(f"Số lượng khuôn mặt phát hiện được: {len(faces)}")

            # Tạo nút để tải xuống ảnh kết quả
            result = Image.fromarray(result_image)
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="Tải xuống ảnh kết quả",
                data=byte_im,
                file_name="result.png",
                mime="image/png"
            )

# Ứng dụng Nhận diện Khuôn mặt với Haar Cascade
def haar_cascade_app():
    st.title("Ứng dụng Nhận diện Khuôn mặt với Haar Cascade")

    st.header("1. Dataset")
    st.write("""
    Dataset bao gồm:
    - Ảnh khuôn mặt: từ thư mục 'faces_and_non_faces_data/faces_24x24'
    - Ảnh không phải khuôn mặt: từ thư mục 'faces_and_non_faces_data/non_faces_24x24'
    """)

    # Hiển thị một số ảnh mẫu từ dataset
    sample_images = [
        cv2.imread('faces_and_non_faces_data/faces_24x24/s1_3.png'),
        cv2.imread('faces_and_non_faces_data/faces_24x24/s1_4.png'),
        cv2.imread('faces_and_non_faces_data/non_faces_24x24/image_26.png'),
        cv2.imread('faces_and_non_faces_data/non_faces_24x24/image_209.png')
    ]

    col1, col2, col3, col4 = st.columns(4)
    for i, image in enumerate(sample_images, 1):
        with locals()[f'col{i}']:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f'Ảnh mẫu {i}', use_column_width=True)

    st.header("2. Giới thiệu phương pháp")
    st.write("""
    Phương pháp nhận diện khuôn mặt này sử dụng Haar Cascade Classifier của OpenCV:

    1. Haar Cascade Classifier:
       - Là một phương pháp học máy dựa trên object detection framework của Viola-Jones.
       - Sử dụng các đặc trưng Haar để phát hiện khuôn mặt trong ảnh.
       - Nhanh và hiệu quả cho việc phát hiện khuôn mặt trong thời gian thực.

    Quy trình:
    1. Chuyển đổi ảnh đầu vào sang ảnh xám.
    2. Áp dụng Haar Cascade Classifier để phát hiện khuôn mặt.
    3. Vẽ hình chữ nhật xung quanh các khuôn mặt được phát hiện.
    """)

    st.header("3. Tham số huấn luyện")
    st.write("""
    Các tham số huấn luyện không thay đổi:
    - Kích thước ảnh đầu vào: 24x24 pixels
    - Loại đặc trưng Haar: edge, line, four-rectangle
    - Tỷ lệ chia tập train/test: 80/20
    - Số ảnh test từ Hugging Face dataset: 50

    Các tham số huấn luyện thay đổi:
    - Giá trị K trong KNN: từ 1 đến 29 (chỉ số lẻ)
    """)

    st.header("4. Kết quả nhận diện")
    st.write("""
    Dưới đây là kết quả nhận diện khuôn mặt sử dụng Haar Cascade Classifier:
    """)

    # Chỗ để thêm hình ảnh
    st.image("Metirc/Figure_1.png", caption="Kết quả nhận diện mẫu", use_column_width=True)
    st.write("""
    - K = 1: Average IoU Score = 0.14810457665471044
    - K = 3: Average IoU Score = 0.17402182875241237
    - K = 5: Average IoU Score = 0.18074834771021372
    - K = 7: Average IoU Score = 0.19036256993243597
    - K = 9: Average IoU Score = 0.193927701365388
    - K = 11: Average IoU Score = 0.213927701365388
    - K = 13: Average IoU Score = 0.213927701365388
    - K = 15: Average IoU Score = 0.193927701365388
    - K = 17: Average IoU Score = 0.193927701365388
    - K = 19: Average IoU Score = 0.202377701365388
    - K = 21: Average IoU Score = 0.2063614703972754
    - K = 23: Average IoU Score = 0.2063614703972754
    - K = 25: Average IoU Score = 0.2162081926194976
    - K = 27: Average IoU Score = 0.2162081926194976
    - K = 29: Average IoU Score = 0.2256774312523117
    - Best K value: 29 with Average IoU Score: 0.2256774312523117
    """)

    st.header("5. Nhận diện khuôn mặt")
    st.write("""
    Tải lên một ảnh để thực hiện nhận diện khuôn mặt:
    """)

    def detect_faces(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Cải thiện chất lượng ảnh
        gray = cv2.equalizeHist(gray)
        
        # Load Haar Cascade Classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Vẽ hình chữ nhật xung quanh khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        return image

    uploaded_image = st.file_uploader("Tải lên ảnh để nhận diện khuôn mặt", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        
        # Nhận diện khuôn mặt
        result_image = detect_faces(image)
        
        # Hiển thị kết quả
        st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption='Kết quả nhận diện', use_column_width=True)

    st.header("6. Kết luận")
    st.write("""
    Phương pháp nhận diện khuôn mặt sử dụng Haar Cascade Classifier có những ưu điểm sau:

    1. Nhanh và hiệu quả cho việc phát hiện khuôn mặt trong thời gian thực.
    2. Không yêu cầu tài nguyên tính toán lớn.
    3. Dễ dàng triển khai và sử dụng với thư viện OpenCV.

    Tuy nhiên, phương pháp này cũng có một số hạn chế:

    1. Có thể gặp khó khăn với các khuôn mặt ở góc nghiêng hoặc bị che khuất một phần.
    2. Độ chính xác có thể không bằng các phương pháp học sâu hiện đại trong một số trường hợp phức tạp.
    3. Nhạy cảm với điều kiện ánh sáng và chất lượng ảnh đầu vào.
    """)

# Ứng dụng Phân đoạn Hình ảnh với GrabCut
def grabcut_app():
    st.title("Ứng Dụng Phân Đoạn Hình Ảnh với GrabCut")

    def resize_image(image, max_width=800, max_height=600):
        """Điều chỉnh kích thước ảnh để vừa với kích thước tối đa cho phép"""
        ratio = min(max_width / image.width, max_height / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.LANCZOS)

    def preprocess_image(image):
        """Tiền xử lý ảnh"""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def grabcut_segmentation(image, rect, mask, iterations=10):
        """Áp dụng thuật toán GrabCut để phân đoạn ảnh"""
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        if rect:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        else:
            cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return mask2

    def postprocess_mask(mask):
        """Hậu xử lý mặt nạ"""
        mask = mask.astype(np.uint8) * 255
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def multi_scale_grabcut(image, rect, mask, scales=[1.0, 0.5, 0.25]):
        """Áp dụng GrabCut ở nhiều tỷ lệ khác nhau"""
        final_mask = np.zeros(mask.shape, dtype=np.uint8)
        for scale in scales:
            scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
            scaled_mask = cv2.resize(mask, None, fx=scale, fy=scale)
            scaled_rect = (int(rect[0]*scale), int(rect[1]*scale), 
                           int(rect[2]*scale), int(rect[3]*scale)) if rect else None
            scaled_mask = grabcut_segmentation(scaled_image, scaled_rect, scaled_mask)
            final_mask += cv2.resize(scaled_mask, mask.shape[::-1]) > 0
        return (final_mask > len(scales) // 2).astype(np.uint8) * 255

    def image_to_bytes(image):
        """Chuyển đổi hình ảnh numpy sang định dạng bytes"""
        img = Image.fromarray(image.astype('uint8'))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Điều chỉnh kích thước ảnh
        image = resize_image(image)
        image_np = np.array(image)
        image_np = preprocess_image(image_np)  # Tiền xử lý ảnh

        # Thêm radio button để chọn chế độ vẽ
        drawing_mode = st.radio("Chọn chế độ vẽ:", ("Hình chữ nhật", "SureForeground", "SureBackground"))
        if drawing_mode == "Hình chữ nhật":
            canvas_drawing_mode = "rect"
            stroke_color = "#000000"
        elif drawing_mode == "SureForeground":
            canvas_drawing_mode = "freedraw"
            stroke_color = "#00FF00"
        else:
            canvas_drawing_mode = "freedraw"
            stroke_color = "#FF0000"

        # Tạo canvas để vẽ
        st.write("Vẽ hình chữ nhật và chọn vùng SureForeground/SureBackground:")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Color for filling the rectangle
            stroke_width=3,
            stroke_color=stroke_color,
            background_image=image,
            drawing_mode=canvas_drawing_mode,
            key="canvas",
            width=image.width,
            height=image.height,
        )

        if st.button("Áp dụng thuật toán GrabCut"):
            if canvas_result.image_data is not None:
                with st.spinner("Đang phân đoạn ảnh, vui lòng đợi..."):
                    mask = np.zeros(image_np.shape[:2], np.uint8) + 2  # Initialize with probable background

                    # Process rectangle
                    rect = None
                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data["objects"]
                        rectangles = [obj for obj in objects if obj["type"] == "rect"]
                        if rectangles:
                            rect_obj = rectangles[-1]
                            rect = (int(rect_obj["left"]), int(rect_obj["top"]), 
                                    int(rect_obj["width"]), int(rect_obj["height"]))
                            mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = 3  # Probable foreground

                    # Process SureForeground and SureBackground
                    mask_data = canvas_result.image_data
                    if mask_data.shape[:2] != mask.shape:
                        mask_data = cv2.resize(mask_data, (mask.shape[1], mask.shape[0]))
                    
                    mask[np.all(mask_data == [0, 255, 0, 255], axis=-1)] = 1  # GC_FGD
                    mask[np.all(mask_data == [255, 0, 0, 255], axis=-1)] = 0  # GC_BGD

                    # Áp dụng multi-scale GrabCut
                    final_mask = multi_scale_grabcut(image_np, rect, mask)
                    
                    # Hậu xử lý mask
                    final_mask = postprocess_mask(final_mask)

                    # Tạo ảnh phân đoạn cuối cùng
                    segmented_image = image_np * (final_mask > 0)[:, :, np.newaxis]

                st.subheader("Kết quả")

                # Hiển thị ảnh gốc và ảnh phân đoạn GrabCut cạnh nhau
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image_np, caption="Ảnh gốc", use_column_width=True)

                with col2:
                    st.image(segmented_image, caption="Ảnh phân đoạn với GrabCut", use_column_width=True)
                    
                # Thêm nút tải về
                segmented_image_bytes = image_to_bytes(segmented_image)
                st.download_button(
                    label="Tải về ảnh phân đoạn",
                    data=segmented_image_bytes,
                    file_name="segmented_image.png",
                    mime="image/png"
                )
            else:
                st.write("Vui lòng vẽ hình chữ nhật hoặc chọn vùng SureForeground/SureBackground.")

# Ứng dụng Xác thực Khuôn mặt Sinh viên
@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

@st.cache_data
def load_embeddings():
    with open('id_card_embeddings.pkl', 'rb') as f:
        return pickle.load(f)

def face_verification_app():
    st.title('Ứng dụng Xác thực Khuôn mặt Sinh viên')

    detector, facenet = load_models()
    id_card_embeddings = load_embeddings()

    def detect_and_align_face(image):
        faces = detector.detect_faces(image)
        if not faces:
            return None
        x, y, width, height = faces[0]['box']
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        return face

    def extract_features(face):
        face = (face.astype('float32') - 127.5) / 128.0
        embedding = facenet.embeddings(np.array([face]))
        return embedding[0]

    def compare_faces(embedding1, embedding2, threshold=0.7):
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity > threshold, similarity

    def verify_face(person_image):
        person_image = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGB2BGR)
        person_face = detect_and_align_face(person_image)
        if  person_face is None:
            return None, 0, "Không thể phát hiện khuôn mặt trong ảnh"
        person_embedding = extract_features(person_face)
        
        best_match = None
        best_similarity = -1
        
        for filename, id_card_embedding in id_card_embeddings.items():
            similarity = cosine_similarity([person_embedding], [id_card_embedding])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = filename
        
        is_match = best_similarity > 0.7  # Có thể điều chỉnh ngưỡng này
        return is_match, best_similarity, best_match

    uploaded_image = st.file_uploader("Tải lên ảnh chân dung của sinh viên", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Ảnh đã tải lên', use_column_width=True)
        
        is_match, similarity, best_match = verify_face(image)
        
        if is_match is None:
            st.error("Không thể phát hiện khuôn mặt trong ảnh.")
        elif is_match:
            st.success(f"Xác thực thành công! Đây là sinh viên trong ảnh: {os.path.splitext(best_match)[0]}")
            st.write(f"Độ tương đồng: {similarity:.2f}")
        else:
            st.warning("Không tìm thấy sinh viên phù hợp trong cơ sở dữ liệu.")
            st.write(f"Độ tương đồng cao nhất: {similarity:.2f}")

        # Hiển thị thông tin chi tiết
        st.subheader("Thông tin chi tiết:")
        st.write(f"Tên file ảnh phù hợp nhất: {best_match}")
        st.write(f"Độ tương đồng: {similarity:.4f}")
        st.write(f"Ngưỡng xác thực: 0.7")

# Ứng dụng chính
def main():
    st.set_page_config(layout="wide")
    
    st.sidebar.title("Chọn ứng dụng")
    
    app_options = {
        "Phân đoạn ảnh với Watershed": watershed_app,
        "Phát hiện Khuôn mặt": face_detection_app,
        "Nhận diện Khuôn mặt với Haar Cascade": haar_cascade_app,
        "Phân đoạn Hình ảnh với GrabCut": grabcut_app,
        "Xác thực Khuôn mặt Sinh viên": face_verification_app
    }
    
    selected_app = st.sidebar.radio("", list(app_options.keys()))
    
    # Xóa nội dung cũ
    st.empty()
    
    # Hiển thị ứng dụng được chọn
    app_options[selected_app]()

if __name__ == "__main__":
    main()