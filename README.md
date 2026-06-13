# BTL_TTNT - Nhận diện cảm xúc khuôn mặt (FER-2013)

## 1. Tổng quan repository

Đây là dự án nhận diện cảm xúc khuôn mặt sử dụng mô hình **ANN (Artificial Neural Network)** huấn luyện trên dữ liệu ảnh FER-2013 đã được tổ chức theo thư mục.

Repository hiện tập trung vào 2 phần chính:
- **Huấn luyện mô hình** từ dữ liệu ảnh (`train_ann.py`)
- **Ứng dụng demo dự đoán** bằng giao diện web với Streamlit (`app.py`)

Ngoài mã nguồn, repository cũng chứa sẵn:
- Mô hình đã huấn luyện (`emotion_ann.keras`, `emotion_ann.h5`)
- Các biểu đồ đánh giá (`accuracy.png`, `loss.png`, `confusion_matrix.png`, `model_summary.png`)

---

## 2. Cấu trúc codebase

```text
BTL_TTNT/
├── FER-2013/
│   ├── train/                  # Dữ liệu train theo lớp cảm xúc
│   ├── test/                   # Dữ liệu test theo lớp cảm xúc
│   ├── train_ann.py            # Script huấn luyện và đánh giá mô hình ANN
│   ├── app.py                  # Ứng dụng Streamlit dự đoán cảm xúc từ ảnh
│   └── evaluate.py             # File dự phòng đánh giá (hiện chưa có nội dung)
├── emotion_ann.keras           # Model đã huấn luyện (định dạng Keras)
├── emotion_ann.h5              # Model đã huấn luyện (định dạng H5)
├── accuracy.png                # Biểu đồ độ chính xác
├── loss.png                    # Biểu đồ loss
├── confusion_matrix.png        # Ma trận nhầm lẫn
└── model_summary.png           # Ảnh tóm tắt kiến trúc model
```

---

## 3. Công nghệ chính được sử dụng

- **Python**
- **TensorFlow / Keras**: xây dựng, huấn luyện và lưu mô hình
- **scikit-learn**: `classification_report`, `confusion_matrix`
- **Matplotlib + Seaborn**: trực quan hóa metric và confusion matrix
- **NumPy**: xử lý mảng dữ liệu ảnh
- **Pillow (PIL)**: đọc và tiền xử lý ảnh upload
- **Streamlit**: tạo giao diện web demo suy luận mô hình

---

## 4. Cách tổ chức mã và luồng xử lý

### 4.1 Huấn luyện (`FER-2013/train_ann.py`)

Script này thực hiện toàn bộ pipeline:
1. Nạp dữ liệu từ `FER-2013/train` và `FER-2013/test` bằng `ImageDataGenerator`
2. Chuẩn hóa ảnh (rescale về `[0, 1]`), ảnh grayscale kích thước `48x48`
3. Xây mô hình ANN dạng `Flatten -> Dense(1024) -> Dense(512) -> Dense(256) -> Dense(7, softmax)`
4. Huấn luyện với:
   - `optimizer="sgd"`
   - `loss="categorical_crossentropy"`
   - `EarlyStopping` theo `val_loss`
5. Đánh giá mô hình trên tập test
6. In báo cáo phân loại (`classification_report`)
7. Vẽ và lưu biểu đồ accuracy/loss + confusion matrix
8. Lưu model ra file `emotion_ann.keras`

### 4.2 Ứng dụng dự đoán (`FER-2013/app.py`)

Ứng dụng Streamlit:
1. Nạp model đã huấn luyện (`emotion_ann.keras`)
2. Cho người dùng upload ảnh khuôn mặt (`jpg/jpeg/png`)
3. Tiền xử lý ảnh giống lúc train:
   - chuyển grayscale
   - resize về `48x48`
   - chuẩn hóa và reshape về tensor đầu vào
4. Dự đoán cảm xúc thuộc 1 trong 7 lớp:
   `angry, disgust, fear, happy, neutral, sad, surprise`
5. Hiển thị nhãn cảm xúc và độ tin cậy

---

## 5. Tóm tắt cách code được tổ chức

- Dự án đang tổ chức theo kiểu **script-based** (mỗi file đảm nhận một luồng công việc chính).
- Chưa tách thành package/module nhiều lớp.
- Dữ liệu được quản lý trực tiếp bằng cấu trúc thư mục theo class label.
- Thành phần suy luận (Streamlit app) phụ thuộc trực tiếp vào model đã huấn luyện và lưu ở root repository.

---

## 6. Gợi ý chạy dự án nhanh

Từ thư mục gốc repository:

1. Huấn luyện mô hình:
   ```bash
   python FER-2013/train_ann.py
   ```
2. Chạy ứng dụng demo:
   ```bash
   streamlit run FER-2013/app.py
   ```

> Lưu ý: cần cài trước các thư viện Python tương ứng (TensorFlow, Streamlit, NumPy, Pillow, scikit-learn, Matplotlib, Seaborn).
