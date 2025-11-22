import numpy as np


def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
    """Tự chia train/test không dùng sklearn."""
    if seed is not None:
        np.random.seed(seed)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-12)


def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-12)


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)


class NumpyLogisticRegression:
    """Logistic Regression implement hoàn toàn bằng NumPy."""

    def __init__(self, lr=0.01, n_iter=2000, l2=0.0, verbose=False):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.verbose = verbose
        self.w = None
        self.b = None
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.astype(np.float64)

        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for i in range(self.n_iter):
            linear_output = X.dot(self.w) + self.b
            y_pred = sigmoid(linear_output)

            dw = (1.0 / n_samples) * X.T.dot(y_pred - y) + (self.l2 * self.w) / n_samples
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if i % 50 == 0 or i == self.n_iter - 1:
                loss = binary_cross_entropy(y, y_pred)
                self.losses.append(loss)
                if self.verbose:
                    print(f"Iter {i:4d} | Loss: {loss:.6f}")

    def predict_proba(self, X):
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


class NumpyKNNClassifier:
    """KNN Classifier đơn giản dùng NumPy.
    Chú ý: độ phức tạp O(N^2), chỉ dùng cho demo / dataset vừa.
    """

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.astype(np.float64)
        self.y_train = y.astype(int)

    def _predict_one(self, x):
        """Dự đoán cho 1 sample x (1D)."""
        # khoảng cách Euclidean tới mọi điểm train (vector hoá)
        diffs = self.X_train - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        # lấy k hàng xóm gần nhất
        idx_sorted = np.argsort(dists)
        k_idx = idx_sorted[: self.k]
        k_labels = self.y_train[k_idx]
        # vote majority
        vals, counts = np.unique(k_labels, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        X = X.astype(np.float64)
        n_samples = X.shape[0]
        preds = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            preds[i] = self._predict_one(X[i])
        return preds


def k_fold_cross_val(model_class, X, y, k_folds=5, **model_kwargs):
    """Cross-validation đơn giản cho binary classification.
    model_class: class, ví dụ NumpyLogisticRegression
    Trả về: dict trung bình các metric.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(k_folds, n_samples // k_folds, dtype=int)
    fold_sizes[: n_samples % k_folds] += 1

    current = 0
    metrics_acc = []
    metrics_f1 = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        metrics_acc.append(acc)
        metrics_f1.append(f1)

        current = stop

    return {
        "accuracy_mean": float(np.mean(metrics_acc)),
        "accuracy_std": float(np.std(metrics_acc)),
        "f1_mean": float(np.mean(metrics_f1)),
        "f1_std": float(np.std(metrics_f1)),
    }
