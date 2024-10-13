import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image
import io

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

def main():
    st.title("Ứng Dụng Phân Đoạn Hình Ảnh với GrabCut")

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

if __name__ == "__main__":
    main()