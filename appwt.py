import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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