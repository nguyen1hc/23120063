import numpy as np

# Các giá trị được coi là missing
MISSING_VALUES = {"", "NA", "NaN", "nan", "?", "NULL", "null"}


# =========================
# 1. LOAD & CƠ BẢN
# =========================

def load_dataset(path, delimiter=",", encoding="utf-8"):
    """
    Đọc file CSV chỉ bằng NumPy.
    Trả về:
        header: list tên cột
        data: np.ndarray (dtype=str), tất cả dữ liệu (không gồm dòng header)
    """
    with open(path, "r", encoding=encoding) as f:
        header = f.readline().strip().split(delimiter)
    data = np.genfromtxt(
        path,
        delimiter=delimiter,
        dtype=str,
        skip_header=1,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def get_column_indices(header, cols):
    name_to_idx = {}
    for i, name in enumerate(header):
        name_to_idx[name] = i
    indices = []
    for c in cols:
        indices.append(name_to_idx[c])
    return indices


def is_missing_array(col):
    def is_miss(x):
        return x in MISSING_VALUES
    vfunc = np.vectorize(is_miss)
    return vfunc(col)


# =========================
# 2. CHUYỂN KIỂU DỮ LIỆU CƠ BẢN
# =========================

def to_float_array(col):
    """
    Chuyển cột string -> float, missing -> np.nan.
    Fix cho NumPy 2.x (tránh DTypePromotionError).
    """
    col = np.asarray(col, dtype=str)
    mask = is_missing_array(col)

    # Ép sang object để có thể gán np.nan
    result = col.astype(object)
    result[mask] = np.nan

    # Cuối cùng convert sang float
    return result.astype(float)


def encode_categorical_to_int(col):
    """
    Mã hoá cột categoric sang integer (0..n_classes-1).
    Trả về:
        int_col: np.ndarray(int)
        mapping: dict {giá trị gốc: id}
    """
    unique_vals, inverse = np.unique(col, return_inverse=True)
    mapping = {}
    for i, v in enumerate(unique_vals):
        mapping[v] = i
    return inverse.astype(int), mapping


def one_hot_encode_from_int(int_col, num_classes=None):
    if num_classes is None:
        num_classes = int_col.max() + 1
    eye = np.eye(num_classes, dtype=float)
    return eye[int_col]


# =========================
# 3. XỬ LÝ MISSING
# =========================

def fill_missing_numeric(col, strategy="mean", fill_value=None):
    col = col.astype(float)
    mask_nan = np.isnan(col)
    if not mask_nan.any():
        return col

    if strategy == "mean":
        value = np.nanmean(col)
    elif strategy == "median":
        value = np.nanmedian(col)
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'constant' strategy")
        value = float(fill_value)
    else:
        raise ValueError("strategy must be 'mean', 'median', or 'constant'")

    col[mask_nan] = value
    return col


def fill_missing_categorical(col, strategy="mode", fill_value=None):
    col = col.astype(str)
    mask_missing = is_missing_array(col)
    if not mask_missing.any():
        return col

    if strategy == "mode":
        vals = col[~mask_missing]
        unique_vals, counts = np.unique(vals, return_counts=True)
        idx_max = np.argmax(counts)
        value = unique_vals[idx_max]
    elif strategy == "constant":
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'constant' strategy")
        value = fill_value
    else:
        raise ValueError("strategy must be 'mode' hoặc 'constant'")

    col[mask_missing] = value
    return col


# =========================
# 4. SCALE / NORMALIZATION
# =========================

def min_max_scale(X):
    """
    Min-Max scaling theo từng cột: (x - min) / (max - min)
    """
    X = X.astype(float)
    min_vals = np.nanmin(X, axis=0)
    max_vals = np.nanmax(X, axis=0)
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1, ranges)
    X_scaled = (X - min_vals) / ranges
    return X_scaled, min_vals, max_vals


def standardize_zscore(X):
    """
    Z-score: (x - mean) / std
    """
    X = X.astype(float)
    means = np.nanmean(X, axis=0)
    stds = np.nanstd(X, axis=0)
    stds = np.where(stds == 0, 1, stds)
    X_std = (X - means) / stds
    return X_std, means, stds


def log_transform(X, epsilon=1e-6):
    """
    Log transform cho dữ liệu dương.
    X có thể là vector hoặc ma trận.

    epsilon thêm vào để tránh log(0).
    Nếu có giá trị <= 0, sẽ dịch toàn bộ cột đó lên.
    """
    X = X.astype(float)
    X_out = X.copy()

    if X_out.ndim == 1:
        min_val = np.nanmin(X_out)
        shift = 0.0
        if min_val <= 0:
            shift = -min_val + epsilon
        X_out = np.log(X_out + shift)
        return X_out, shift
    else:
        shifts = []
        for j in range(X_out.shape[1]):
            col = X_out[:, j]
            min_val = np.nanmin(col)
            shift = 0.0
            if min_val <= 0:
                shift = -min_val + epsilon
            X_out[:, j] = np.log(col + shift)
            shifts.append(shift)
        return X_out, np.array(shifts)


def decimal_scaling(X):
    """
    Decimal scaling:
        x' = x / 10^d
    với d là số chữ số của max(|x|) trên mỗi cột.
    """
    X = X.astype(float)
    X_out = X.copy()

    if X_out.ndim == 1:
        max_abs = np.nanmax(np.abs(X_out))
        if max_abs == 0:
            d = 0
        else:
            d = int(np.ceil(np.log10(max_abs)))
        X_out = X_out / (10 ** d)
        return X_out, d
    else:
        ds = []
        for j in range(X_out.shape[1]):
            col = X_out[:, j]
            max_abs = np.nanmax(np.abs(col))
            if max_abs == 0:
                d = 0
            else:
                d = int(np.ceil(np.log10(max_abs)))
            X_out[:, j] = col / (10 ** d)
            ds.append(d)
        return X_out, np.array(ds)


# =========================
# 5. OUTLIER (IQR)
# =========================

def iqr_outlier_mask(col, factor=1.5):
    """
    Tạo mask đánh dấu OUTLIER (True nếu là outlier) cho 1 cột numeric theo IQR.

    Q1, Q3 = percentiles 25%, 75%
    IQR = Q3 - Q1
    outlier nếu < Q1 - factor*IQR hoặc > Q3 + factor*IQR
    """
    col = col.astype(float)
    q1 = np.nanpercentile(col, 25)
    q3 = np.nanpercentile(col, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    mask = (col < lower) | (col > upper)
    mask = np.where(np.isnan(col), False, mask)
    return mask, (lower, upper)


def remove_outliers_iqr(X, factor=1.5):
    """
    Loại bỏ hàng chứa outlier (theo IQR) trên bất kỳ cột nào.

    Trả về:
        X_clean, mask_keep
    """
    X = X.astype(float)
    n_samples, n_features = X.shape
    mask_keep = np.ones(n_samples, dtype=bool)

    for j in range(n_features):
        col = X[:, j]
        out_mask, bounds = iqr_outlier_mask(col, factor=factor)
        mask_keep = mask_keep & (~out_mask)

    return X[mask_keep], mask_keep


# =========================
# 6. BUILD X, y (PIPELINE CƠ BẢN)
# =========================

def build_feature_matrix(
    header,
    data,
    numeric_cols,
    categorical_cols,
    label_col,
    numeric_missing_strategy="mean",
    categorical_missing_strategy="mode",
):
    """
    Từ raw data -> X (float), y (int 0/1), kèm metadata (dict).
    - numeric_cols: danh sách cột numeric (float, int)
    - categorical_cols: danh sách cột categorical (sẽ one-hot)
        (tất cả xử lý giống nhau, không đặc biệt 'experience' nữa)
    """
    # map tên cột -> index
    col_idx = {}
    for i, name in enumerate(header):
        col_idx[name] = i

    # ----- LABEL -----
    y_raw = data[:, col_idx[label_col]].astype(str)
    mask_missing_label = is_missing_array(y_raw)
    if mask_missing_label.any():
        data = data[~mask_missing_label]
        y_raw = y_raw[~mask_missing_label]

    y_int, label_mapping = encode_categorical_to_int(y_raw)
    if len(label_mapping) != 2:
        raise ValueError(
            "Label column '%s' không phải nhị phân. Mapping: %s"
            % (label_col, str(label_mapping))
        )

    # ----- NUMERIC FEATURES -----
    X_num_list = []
    numeric_info = {}
    for col_name in numeric_cols:
        col = data[:, col_idx[col_name]]
        col_float = to_float_array(col)
        col_filled = fill_missing_numeric(col_float, strategy=numeric_missing_strategy)
        X_num_list.append(col_filled.reshape(-1, 1))
        numeric_info[col_name] = {"missing_strategy": numeric_missing_strategy}

    if len(X_num_list) > 0:
        X_num = np.hstack(X_num_list)
    else:
        X_num = np.empty((data.shape[0], 0))

    # ----- CATEGORICAL FEATURES (ONE-HOT) -----
    X_cat_list = []
    cat_info = {}

    for col_name in categorical_cols:
        col = data[:, col_idx[col_name]].astype(str)

        col_filled = fill_missing_categorical(col, strategy=categorical_missing_strategy)
        int_col, mapping = encode_categorical_to_int(col_filled)
        one_hot = one_hot_encode_from_int(int_col)
        X_cat_list.append(one_hot)
        cat_info[col_name] = {
            "type": "nominal",
            "mapping": mapping,
            "missing_strategy": categorical_missing_strategy,
            "num_classes": len(mapping),
        }

    if len(X_cat_list) > 0:
        X_cat = np.hstack(X_cat_list)
    else:
        X_cat = np.empty((data.shape[0], 0))

    # ----- GHÉP NUMERIC + CATEGORICAL -----
    X = np.hstack([X_num, X_cat])

    metadata = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "label_col": label_col,
        "label_mapping": label_mapping,
        "numeric_info": numeric_info,
        "categorical_info": cat_info,
    }

    return X.astype(float), y_int.astype(int), metadata


# =========================
# 7. TRAIN/TEST SPLIT
# =========================

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test


# =========================
# 8. THỐNG KÊ MÔ TẢ
# =========================

def describe_numeric(X):
    X = X.astype(float)
    desc = {}
    desc["mean"] = np.nanmean(X, axis=0)
    desc["std"] = np.nanstd(X, axis=0)
    desc["min"] = np.nanmin(X, axis=0)
    desc["25%"] = np.nanpercentile(X, 25, axis=0)
    desc["50%"] = np.nanpercentile(X, 50, axis=0)
    desc["75%"] = np.nanpercentile(X, 75, axis=0)
    desc["max"] = np.nanmax(X, axis=0)
    return desc
