import numpy as np

class LinearRegression:
    """Linear Regression implement từ đầu bằng NumPy"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _mean_squared_error(self, y_true, y_pred):
        """Hàm mất mát MSE"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_gradients(self, X, y, y_pred):
        """Tính gradients cho Linear Regression"""
        n_samples = X.shape[0]
        dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
        db = (-2 / n_samples) * np.sum(y - y_pred)
        return dw, db
    
    def fit(self, X, y):
        """Huấn luyện mô hình bằng Gradient Descent"""
        n_samples, n_features = X.shape
        
        # Khởi tạo parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Dự đoán
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Tính loss
            loss = self._mean_squared_error(y, y_pred)
            self.losses.append(loss)
            
            # Tính gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Cập nhật parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Dự đoán"""
        return np.dot(X, self.weights) + self.bias


class LogisticRegression:
    """Logistic Regression implement từ đầu bằng NumPy"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        """Hàm sigmoid"""
        z = np.clip(z, -500, 500)  # Ổn định số học
        return 1 / (1 + np.exp(-z))
    
    def _binary_cross_entropy(self, y_true, y_pred):
        """Hàm mất mát Binary Cross-Entropy"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _compute_gradients(self, X, y, y_pred):
        """Tính gradients cho Logistic Regression"""
        n_samples = X.shape[0]
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        return dw, db
    
    def fit(self, X, y):
        """Huấn luyện mô hình"""
        n_samples, n_features = X.shape
        
        # Khởi tạo parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            
            # Tính loss
            loss = self._binary_cross_entropy(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Cập nhật parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        """Dự đoán xác suất"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)
    
    def predict(self, X, threshold=0.5):
        """Dự đoán nhãn"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)


class KNN:
    """K-Nearest Neighbors implement từ đầu bằng NumPy"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1, x2):
        """Tính khoảng cách Euclidean"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
    
    def fit(self, X, y):
        """Lưu dữ liệu training"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Dự đoán nhãn"""
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Tính khoảng cách tới tất cả điểm training
            distances = self._euclidean_distance(X[i], self.X_train)
            
            # Lấy k hàng xóm gần nhất
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Bỏ phiếu đa số
            y_pred[i] = np.bincount(k_nearest_labels.astype(int)).argmax()
        
        return y_pred


class GradientDescentOptimizer:
    """Thuật toán tối ưu Gradient Descent implement từ đầu"""
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
    
    def initialize_velocity(self, parameters):
        """Khởi tạo velocity cho momentum"""
        self.velocity = [np.zeros_like(param) for param in parameters]
    
    def update(self, parameters, gradients):
        """Cập nhật parameters với momentum"""
        if self.velocity is None:
            self.initialize_velocity(parameters)
        
        updated_parameters = []
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            updated_parameters.append(param + self.velocity[i])
        
        return updated_parameters


class EvaluationMetrics:
    """Các độ đo đánh giá implement từ đầu bằng NumPy"""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Độ chính xác"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true, y_pred):
        """Precision"""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return true_positives / (predicted_positives + 1e-9)
    
    @staticmethod
    def recall(y_true, y_pred):
        """Recall"""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return true_positives / (actual_positives + 1e-9)
    
    @staticmethod
    def f1_score(y_true, y_pred):
        """F1-Score"""
        prec = EvaluationMetrics.precision(y_true, y_pred)
        rec = EvaluationMetrics.recall(y_true, y_pred)
        return 2 * (prec * rec) / (prec + rec + 1e-9)
    
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """MSE cho regression"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mean_absolute_error(y_true, y_pred):
        """MAE cho regression"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """R-squared cho regression"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-9))
    
    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """Ma trận nhầm lẫn"""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def classification_report(y_true, y_pred):
        """Báo cáo phân loại đầy đủ"""
        return {
            'accuracy': EvaluationMetrics.accuracy(y_true, y_pred),
            'precision': EvaluationMetrics.precision(y_true, y_pred),
            'recall': EvaluationMetrics.recall(y_true, y_pred),
            'f1_score': EvaluationMetrics.f1_score(y_true, y_pred),
            'confusion_matrix': EvaluationMetrics.confusion_matrix(y_true, y_pred)
        }


class CrossValidation:
    """Cross-validation implement từ đầu bằng NumPy"""
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=42):
        """Chia dữ liệu thành train/test"""
        np.random.seed(random_state)
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        # Xáo trộn indices
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    @staticmethod
    def k_fold_split(X, y, k=5, random_state=42):
        """Chia dữ liệu thành k folds"""
        np.random.seed(random_state)
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        
        fold_size = n_samples // k
        folds = []
        
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n_samples
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            folds.append((train_indices, val_indices))
        
        return folds
    
    @staticmethod
    def cross_val_score(model, X, y, cv=5, scoring='accuracy', random_state=42):
        """Thực hiện k-fold cross validation"""
        folds = CrossValidation.k_fold_split(X, y, k=cv, random_state=random_state)
        scores = []
        
        for train_idx, val_idx in folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Huấn luyện mô hình
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            
            # Dự đoán
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_val)
            elif hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                raise ValueError("Model must have predict or predict_proba method")
            
            # Tính điểm
            if scoring == 'accuracy':
                score = EvaluationMetrics.accuracy(y_val, y_pred)
            elif scoring == 'precision':
                score = EvaluationMetrics.precision(y_val, y_pred)
            elif scoring == 'recall':
                score = EvaluationMetrics.recall(y_val, y_pred)
            elif scoring == 'f1':
                score = EvaluationMetrics.f1_score(y_val, y_pred)
            elif scoring == 'mse':
                score = EvaluationMetrics.mean_squared_error(y_val, y_pred)
            elif scoring == 'r2':
                score = EvaluationMetrics.r2_score(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported scoring: {scoring}")
            
            scores.append(score)
        
        return np.array(scores)


class StandardScaler:
    """Chuẩn hóa dữ liệu implement từ đầu"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        """Tính mean và std"""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1, self.std)  # Tránh chia 0
    
    def transform(self, X):
        """Chuẩn hóa dữ liệu"""
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit và transform"""
        self.fit(X)
        return self.transform(X)