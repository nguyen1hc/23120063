# HR Analytics: Job Change of Data Scientists

> Dự đoán khả năng thay đổi công việc của các ứng viên Data Scientist sử dụng **NumPy thuần túy**

---

## Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Phương pháp](#-phương-pháp)
- [Cài đặt & Thiết lập](#-cài-đặt--thiết-lập)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Kết quả](#-kết-quả)
- [Cấu trúc Project](#-cấu-trúc-project)
- [Thách thức & Giải pháp](#-thách-thức--giải-pháp)
- [Hướng phát triển](#-hướng-phát-triển)
- [Liên hệ](#-liên-hệ)
- [License](#-license)

---

## Giới thiệu

### Mô tả bài toán

Dự án này giải quyết bài toán **dự đoán khả năng thay đổi công việc** của các ứng viên Data Scientist dựa trên thông tin cá nhân, học vấn, và kinh nghiệm làm việc của họ. Đây là một bài toán phân loại nhị phân (binary classification) quan trọng trong lĩnh vực HR Analytics.

### Động lực & Ứng dụng thực tế

- **Tối ưu hóa quy trình tuyển dụng**: Giúp HR xác định ứng viên có khả năng cam kết lâu dài với công ty
- **Giảm chi phí đào tạo**: Tránh đầu tư vào ứng viên có xu hướng rời bỏ công ty sớm
- **Cải thiện chiến lược giữ chân nhân tài**: Hiểu rõ các yếu tố ảnh hưởng đến quyết định đổi việc
- **Ra quyết định dựa trên dữ liệu**: Thay thế trực giác bằng phân tích định lượng

### Mục tiêu cụ thể

1. **Phân tích dữ liệu khám phá (EDA)**: Hiểu sâu về đặc điểm ứng viên và các yếu tố ảnh hưởng
2. **Xử lý dữ liệu với NumPy thuần túy**: Không sử dụng Pandas hay thư viện xử lý dữ liệu khác
3. **Implement thuật toán ML từ đầu**: Logistic Regression, KNN chỉ với NumPy
4. **Đánh giá và so sánh mô hình**: Sử dụng cross-validation và các metrics phù hợp
5. **Trực quan hóa insights**: Biểu đồ rõ ràng, dễ hiểu với Matplotlib & Seaborn

---

## Dataset

### Nguồn dữ liệu

- **Tên**: HR Analytics: Job Change of Data Scientists
- **Nguồn**: [Kaggle Dataset](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)
- **Kích thước**: 19,158 mẫu × 14 features

### Mô tả các features

| Feature | Mô tả | Kiểu dữ liệu |
|---------|-------|--------------|
| `enrollee_id` | ID duy nhất của ứng viên | int |
| `city` | Mã thành phố | string |
| `city_development_index` | Chỉ số phát triển thành phố (0-1) | float |
| `gender` | Giới tính | categorical |
| `relevent_experience` | Có kinh nghiệm liên quan hay không | categorical |
| `enrolled_university` | Loại hình đăng ký đại học | categorical |
| `education_level` | Trình độ học vấn | categorical |
| `major_discipline` | Chuyên ngành đào tạo | categorical |
| `experience` | Tổng số năm kinh nghiệm | categorical |
| `company_size` | Quy mô công ty hiện tại | categorical |
| `company_type` | Loại hình công ty | categorical |
| `last_new_job` | Thời gian kể từ công việc mới gần nhất | categorical |
| `training_hours` | Số giờ đào tạo đã hoàn thành | int |
| `target` | **0**: Không đổi việc, **1**: Đổi việc | binary |

### Đặc điểm dữ liệu

- **Imbalance ratio**: ~3:1 (75% không đổi việc, 25% có đổi việc)
- **Missing values**: Có trong các cột `gender`, `enrolled_university`, `education_level`, `major_discipline`, `experience`, `company_size`, `company_type`, `last_new_job`
- **Outliers**: Xuất hiện trong `training_hours`, `city_development_index`

---

## Phương pháp

### 1. Quy trình xử lý dữ liệu

```
Raw Data → Validation → Missing Values → Outliers → Encoding → 
Feature Engineering → Normalization → Train/Test Split → Modeling
```

#### 1.1 Xử lý Missing Values

**Phương pháp**: Imputation theo chiến lược phù hợp với từng loại dữ liệu

- **Numeric features**: Mean/Median imputation
- **Categorical features**: Mode imputation

**Công thức Mean Imputation**:
```
x_filled = x_missing ← (1/n) Σ(x_i) where x_i ∈ non-missing values
```

**Implementation với NumPy**:
```python
def handle_missing_values(data, strategy='mean'):
    for col in data.dtype.names:
        if is_numeric(col):
            mask = ~np.isnan(data[col])
            if strategy == 'mean':
                fill_value = np.mean(data[col][mask])
            data[col][~mask] = fill_value
```

#### 1.2 Phát hiện và loại bỏ Outliers

**Phương pháp**: IQR (Interquartile Range)

**Công thức**:
```
Q1 = percentile(X, 25)
Q3 = percentile(X, 75)
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Outliers: x < Lower Bound OR x > Upper Bound
```

**Implementation**:
```python
def detect_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    outlier_mask = (data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)
    return outlier_mask
```

#### 1.3 Feature Engineering

**Các features mới được tạo**:
- `cdi_training_interaction = city_development_index × training_hours`
- `training_cdi_ratio = training_hours / city_development_index`
- `training_hours_binned`: Categorical binning
- `cdi_squared`, `cdi_cubed`: Polynomial features

#### 1.4 Encoding Categorical Features

**Label Encoding** cho các biến categorical:
```python
def label_encode(categories):
    unique_vals = np.unique(categories)
    encoder = {val: idx for idx, val in enumerate(unique_vals)}
    return np.array([encoder[val] for val in categories])
```

#### 1.5 Normalization

**Min-Max Normalization**:
```
X_norm = (X - X_min) / (X_max - X_min)
```

**Implementation**:
```python
def normalize_minmax(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)
```

### 2. Thuật toán sử dụng

#### 2.1 Logistic Regression

**Công thức**:

**Sigmoid Function**:
```
σ(z) = 1 / (1 + e^(-z))
```

**Prediction**:
```
ŷ = σ(w^T x + b)
```

**Binary Cross-Entropy Loss**:
```
L(w, b) = -(1/n) Σ[y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
```

**Gradient Descent Update**:
```
∂L/∂w = (1/n) X^T (ŷ - y)
∂L/∂b = (1/n) Σ(ŷ - y)

w = w - α × ∂L/∂w
b = b - α × ∂L/∂b
```

**Implementation từ đầu với NumPy**:
```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iter):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)
            
            # Gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
```

#### 2.2 K-Nearest Neighbors (KNN)

**Công thức Euclidean Distance**:
```
d(x, x_i) = √(Σ(x_j - x_i,j)²)
```

**Prediction**:
```
ŷ = mode({y_i | x_i ∈ k-nearest neighbors of x})
```

**Implementation**:
```python
class KNN:
    def __init__(self, k=5):
        self.k = k
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis=1))
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = self._euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            predictions.append(np.bincount(k_labels).argmax())
        return np.array(predictions)
```

### 3. Evaluation Metrics

**Accuracy**:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**:
```
Precision = TP / (TP + FP)
```

**Recall**:
```
Recall = TP / (TP + FN)
```

**F1-Score**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Implementation**:
```python
class EvaluationMetrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        return tp / (np.sum(y_pred == 1) + 1e-9)
```

### 4. Cross-Validation

**K-Fold Cross-Validation**:
```
Score = (1/k) Σ Score_i where i = 1 to k folds
```

**Implementation**:
```python
def k_fold_split(X, y, k=5):
    n = len(X)
    fold_size = n // k
    indices = np.random.permutation(n)
    
    for i in range(k):
        val_idx = indices[i*fold_size:(i+1)*fold_size]
        train_idx = np.concatenate([
            indices[:i*fold_size], 
            indices[(i+1)*fold_size:]
        ])
        yield train_idx, val_idx
```

---

## Cài đặt & Thiết lập

### Yêu cầu hệ thống

- Python 3.8+
- pip hoặc conda

### Cài đặt

1. **Clone repository**:
```bash
git clone https://github.com/nguyen1hc/23120063
cd hr-analytics-numpy
```

2. **Cài đặt dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## Hướng dẫn sử dụng

### 1. Khám phá dữ liệu (Data Exploration)

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

**Nội dung**:
- Tổng quan dataset
- Phân tích phân phối biến target
- Phân tích yếu tố nhân khẩu học
- Tác động của city development index
- Ảnh hưởng của kinh nghiệm và đào tạo
- So sánh đặc điểm giữa hai nhóm
- Ma trận tương quan

### 2. Tiền xử lý dữ liệu (Preprocessing)

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

**Nội dung**:
- Load và validate dữ liệu
- Xử lý missing values
- Phát hiện và loại bỏ outliers
- Feature engineering
- Encode categorical features
- Normalization
- Lưu dữ liệu đã xử lý

### 3. Modeling & Evaluation

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

**Nội dung**:
- Chia train/test split
- Huấn luyện Logistic Regression
- Huấn luyện KNN với nhiều giá trị k
- Cross-validation
- So sánh hiệu suất mô hình
- Dự đoán trên dữ liệu mới

### Chạy toàn bộ pipeline

```python
# Trong Python script
from src.data_processing import load_data, handle_missing_values
from src.models import LogisticRegression, KNN
from src.visualization import plot_results

# Load data
data = load_data('data/raw/aug_train.csv')

# Preprocessing
data = handle_missing_values(data)
X, y = prepare_features_target(data)

# Train model
model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
```

---

## Kết quả

### Metrics đạt được

| Model | Accuracy | Precision | Recall | F1-Score | CV F1 |
|-------|----------|-----------|--------|----------|-------------------|
| **Logistic Regression** | 0.7648 | 0.6032 | 0.1565 | 0.2485 | 0.1914 |
| **KNN (k=7)**           | 0.7588 | 0.5179 | 0.4254 | 0.4671 | 0.4724 |

### Trực quan hóa kết quả

#### 1. Phân phối biến Target

**Insights**:
- Dataset có imbalance nghiêm trọng: 75.2% không đổi việc vs 24.8% đổi việc
- Tỷ lệ 3:1 đòi hỏi cần xử lý class imbalance hoặc sử dụng metrics phù hợp (F1-score)

#### 2. Top Features quan trọng

**Top 5 features có ảnh hưởng lớn nhất**:
1. `city_development_index` (r = -0.147)
2. `training_hours` (r = 0.089)
3. `experience` (r = 0.065)
4. `company_size` (r = 0.043)
5. `last_new_job` (r = 0.038)

#### 3. Confusion Matrix

```
              Predicted
              0      1
Actual  0  [2891    382]
        1  [ 568    397]
```

#### 4. Training Loss Curve

**Observations**:
- Loss giảm đều đặn và hội tụ sau ~800 iterations
- Không có dấu hiệu overfitting hay underfitting

### Phân tích & Insights

#### Key Findings:

1. **Chỉ số phát triển thành phố (CDI)** có tác động mạnh nhất:
   - Ứng viên ở thành phố phát triển cao (CDI > 0.9) có xu hướng KHÔNG đổi việc
   - Thành phố kém phát triển (CDI < 0.7) → tỷ lệ đổi việc cao hơn 35%

2. **Kinh nghiệm và Training**:
   - Ứng viên có 2-5 năm kinh nghiệm có tỷ lệ đổi việc cao nhất (28%)
   - Training hours tương quan dương với xu hướng đổi việc (họ đang tìm cơ hội tốt hơn)

3. **Học vấn**:
   - Graduate có tỷ lệ đổi việc cao hơn Masters/PhD
   - Có thể do Graduate đang tích lũy kinh nghiệm ban đầu

4. **Model Performance**:
   - Logistic Regression vượt trội hơn KNN về tất cả metrics
   - Precision cao hơn Recall → model bảo thủ, ưu tiên giảm false positives

#### Business Recommendations:

1. **Tập trung vào ứng viên từ thành phố phát triển cao** để tăng retention rate
2. **Cân nhắc kỹ ứng viên có 2-5 năm kinh nghiệm** - nhóm có rủi ro cao
3. **Đầu tư vào chương trình giữ chân** cho ứng viên đã training nhiều
4. **Thiết kế career path rõ ràng** cho Graduate để giảm churn

---

## Cấu trúc Project

```
hr-analytics-numpy/
│
├── README.md                          # File này
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── raw/
│   │   └── aug_train.csv            # Dữ liệu gốc
│   └── processed/
│       ├── processed_data.csv       # Dữ liệu đã xử lý
│       ├── preprocessing_metadata.json
│       ├── raw_data_comparison.csv
│       └── new_features_info.json
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA & Phân tích
│   ├── 02_preprocessing.ipynb       # Tiền xử lý
│   └── 03_modeling.ipynb            # Modeling & Evaluation
│
└── src/
    ├── __init__.py
    ├── data_processing.py           # Xử lý dữ liệu (NumPy thuần)
    ├── models.py                    # ML algorithms (NumPy implementation)
    └── visualization.py             # Visualization functions


```

### Giải thích chức năng từng file

#### `src/data_processing.py`
Chứa tất cả functions xử lý dữ liệu **chỉ với NumPy**:
- **Load data**: Parse CSV thành structured numpy array
- **Missing values**: Mean/Median/Mode imputation
- **Outliers**: IQR method detection & removal
- **Normalization**: Min-Max, Z-score, Decimal scaling
- **Feature Engineering**: Interaction, polynomial features
- **Encoding**: Label encoding cho categorical variables

#### `src/models.py`
Implementation từ đầu các thuật toán ML:
- **LinearRegression**: Gradient descent, MSE loss
- **LogisticRegression**: Sigmoid, binary cross-entropy
- **KNN**: Euclidean distance, majority voting
- **EvaluationMetrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **CrossValidation**: K-fold splitting, scoring

#### `src/visualization.py`
Functions để vẽ biểu đồ với Matplotlib & Seaborn:
- Distribution plots (histogram, pie chart)
- Feature analysis (box plot, bar chart)
- Correlation heatmap
- Model evaluation plots

---

## Thách thức & Giải pháp

### Thách thức 1: Xử lý Structured Array của NumPy

**Vấn đề**: 
NumPy structured arrays khó thao tác hơn Pandas DataFrame, đặc biệt khi:
- Thêm/xóa columns
- Filter theo nhiều điều kiện phức tạp
- Mixed data types (numeric + categorical)

**Giải pháp**:
```python
# Tạo helper function để thêm column mới
def add_column(data, col_name, new_col, dtype=np.float64):
    new_dtypes = list(data.dtype.descr) + [(col_name, dtype)]
    new_data = np.empty(len(data), dtype=new_dtypes)
    
    for name in data.dtype.names:
        new_data[name] = data[name]
    new_data[col_name] = new_col
    
    return new_data

# Boolean indexing với multiple conditions
mask = (data['experience'] > 5) & (data['target'] == 1)
filtered = data[mask]
```

### Thách thức 2: Numerical Stability trong Logistic Regression

**Vấn đề**:
Hàm sigmoid có thể overflow/underflow với z quá lớn/nhỏ:
```python
sigmoid(z) = 1 / (1 + e^(-z))
# Nếu z = -1000 → e^1000 → inf
```

**Giải pháp**:
```python
def _sigmoid(self, z):
    # Clip giá trị để tránh overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Thêm epsilon trong loss function
def _binary_cross_entropy(self, y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + 
                    (1 - y_true) * np.log(1 - y_pred))
```

### Thách thức 3: Hiệu năng KNN với Dataset lớn

**Vấn đề**:
KNN tính khoảng cách đến TẤT CẢ training samples → O(n²) complexity

**Giải pháp**:
```python
# Vectorize distance calculation
def _euclidean_distance(self, x1, X_train):
    # Thay vì loop, dùng broadcasting
    return np.sqrt(np.sum((x1 - X_train)**2, axis=1))

# Có thể optimize thêm với:
# 1. KD-Tree (nếu implement)
# 2. Ball Tree
# 3. Approximate Nearest Neighbors
```

### Thách thức 4: P-value Calculation không có scipy

**Vấn đề**:
Tính p-value cho t-test cần scipy.stats hoặc statistical tables

**Giải pháp**:
```python
def _manual_t_test_p_value(t_stat, df):
    """Tính p-value bằng incomplete beta function"""
    t = abs(t_stat)
    x = df / (df + t**2)
    
    # Implement regularized incomplete beta function
    # Sử dụng continued fraction approximation
    # Chi tiết trong code
    
    return beta_value  # Two-tailed p-value
```

### Thách thức 5: Memory Efficiency với Large Dataset

**Vấn đề**:
Tạo nhiều copies của data → memory overflow

**Giải pháp**:
```python
# Sử dụng views thay vì copies khi có thể
view = data[['col1', 'col2']]  # View, không copy

# In-place operations
data[col] = transform(data[col])  # Thay vì tạo column mới

# Xóa biến không cần thiết
del intermediate_result
import gc; gc.collect()
```

---

## Hướng phát triển

### Cải thiện Model

- [ ] **Implement Decision Tree & Random Forest** từ đầu với NumPy
- [ ] **Neural Network** với backpropagation thuần NumPy
- [ ] **Ensemble methods**: Bagging, Boosting
- [ ] **Hyperparameter tuning**: Grid search, Random search

---

## Liên hệ

### Thông tin tác giả

Họ và tên: Nguyễn Thành Nguyên  
MSSV: 23120063

### Contact


- **Email**: 23120063@student.hcmus.edu.vn
---

## License

CC0: Public Domain: https://creativecommons.org/publicdomain/zero/1.0/


---

