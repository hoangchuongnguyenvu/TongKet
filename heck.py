import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN
from PIL import Image
import io

# Khởi tạo MTCNN
@st.cache_resource
def load_detector():
    return MTCNN()

detector = load_detector()

def detect_faces(image):
    # Phát hiện khuôn mặt bằng MTCNN
    faces = detector.detect_faces(image)
    return faces

def main():
    st.title("Ứng dụng Phát hiện Khuôn mặt")

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

if __name__ == "__main__":
    main()