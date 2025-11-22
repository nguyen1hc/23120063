
# 🔍 Dự đoán Khả Năng Thay Đổi Công Việc Của Ứng Viên Data Science

## Mô tả ngắn gọn về project
Dự án sử dụng NumPy để xử lý dữ liệu và xây dựng mô hình dự đoán khả năng một ứng viên Data Science có muốn thay đổi công việc hay không. Dữ liệu được trực quan hóa bằng Matplotlib/Seaborn và mô hình Logistic Regression + KNN được triển khai **từ đầu, không dùng sklearn**.
github: https://github.com/nguyen1hc/23120063

---

## Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dataset](#dataset)
- [Method](#method)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Challenges & Solutions](#challenges--solutions)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)
- [Thông tin tác giả](#thông-tin-tác-giả)
- [Contact](#contact)
- [License](#license)

---

## Giới thiệu
### **Mô tả bài toán**
Mục tiêu là dự đoán biến nhị phân:
| Giá trị | Ý nghĩa |
|--------|---------|
| `1` | Ứng viên có xu hướng đổi việc |
| `0` | Ứng viên tiếp tục công việc hiện tại |

### **Động lực**
- Hỗ trợ tuyển dụng chiến lược
- Phân tích thị trường nhân lực Data Science
- Tối ưu chi phí tuyển dụng và đào tạo

### **Mục tiêu cụ thể**
- Xử lý dữ liệu hoàn toàn bằng NumPy
- Trực quan hóa xu hướng dữ liệu
- Xây dựng mô hình học máy thủ công
- Đánh giá mô hình bằng các metric tiêu chuẩn
- Tự cài đặt cross-validation

---

## Dataset
### **Nguồn dữ liệu**
Kaggle – *HR Analytics: Job Change of Data Scientists*

### **Đặc điểm dữ liệu**
- ~19k dòng
- Nhiều missing values
- Nhiều biến phân loại → one-hot → ma trận lớn
- Nhiều giá trị không chuẩn như `<1`, `>20`, `never`

### **Các nhóm thuộc tính chính**
| Thuộc tính | Loại | Ví dụ |
|------------|------|-------|
| Nhân khẩu học | categorical | gender, education_level |
| Kinh nghiệm | ordinal | experience, relevant_experience |
| Công ty | categorical | company_type, company_size |
| Numeric | continuous | training_hours, cdi |

---

## Method
### **Quy trình xử lý dữ liệu (NumPy-only)**

| Bước | Mô tả | Công cụ |
|------|------|----------|
| Load dữ liệu | Không dùng pandas | `np.loadtxt()` |
| Missing values | mean/median/mode | Tự cài |
| Encode | one-hot thủ công | NumPy |
| Standardize / Normalize | Z-score, min-max | NumPy |
| Outliers | IQR clipping | NumPy |

### **Thuật toán**
#### Logistic Regression
\[
\hat{y} = \sigma(w^T x + b)
\]

#### KNN
\[
d = \sqrt{\sum (x_i - x_j)^2}
\]
```python
dists = np.sqrt(np.sum((X_train - x)**2, axis=1))
```

#### Cross-validation (Tự cài)
```python
scores = k_fold_cross_val(
    NumpyLogisticRegression,
    X, y, k_folds=5
)
```

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

```bash
cd notebooks
jupyter notebook
```

---

## Usage

### **Khám phá dữ liệu**
```
notebooks/01_data_exploration.ipynb
```

### **Tiền xử lý**
```bash
!jupyter notebook notebooks/02_preprocessing.ipynb
```

### **Huấn luyện mô hình**
```python
from src.models import NumpyLogisticRegression
model = NumpyLogisticRegression(lr=0.1, n_iter=2000)
model.fit(X_train, y_train)
```

---

## Results

### **Logistic Regression**
| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|----|
| Train | 0.7758 | 0.6013 | 0.3012 | 0.4013 |
| Test  | 0.7763 | 0.5908 | 0.3246 | 0.4190 |

### **KNN (k = 5)**
| Dataset | Accuracy | Precision | Recall | F1 |
| Test | 0.7413 | 0.4747 | 0.3845 | 0.4248 |

### **5-Fold Cross Validation**
```
accuracy_mean: 0.7717
accuracy_std : 0.0070
f1_mean      : 0.3690
f1_std       : 0.0073
```

---

## Project Structure

```bash
src/
│── data_processing.py  
│── visualization.py    
│── models.py           
└── __init__.py
```

---

## Challenges & Solutions

| Vấn đề | Nguyên nhân | Giải pháp |
|--------|------------|-----------|
| Không Pandas | Yêu cầu đề | `np.loadtxt()` |
| Không sklearn CV | Muốn điểm cao | tự cài k-fold |
| Overflow sigmoid | logit lớn | `np.clip()` |
| experience không chuẩn | `<1`, `>20` | mapping ordinal |

---

## Future Improvements
- Oversampling lớp 1
- PCA giảm chiều
- Benchmark thêm thuật toán nâng cao
- API dự đoán ứng viên thật

---

## Contributors
- Nguyễn Thành Nguyên

---

## Thông tin tác giả
Họ và tên: Nguyễn Thành Nguyên
MSSV: 23120063

---

## Contact
Email: 23120063@student.hcmus.edu.vn

---

## License
CC0: Public Domain: https://creativecommons.org/publicdomain/zero/1.0/
