import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

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