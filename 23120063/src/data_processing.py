import numpy as np

def load_data(file_path):
    """Load CSV as structured array using only NumPy."""
    try:
        print(f"Loading data from: {file_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            raise ValueError("File is empty")
        
        # Process header
        header = [col.strip().replace('"', '') for col in lines[0].strip().split(',')]
        data_lines = [line.strip() for line in lines[1:] if line.strip()]
        
        print(f"Found {len(data_lines)} data rows with {len(header)} columns")
        
        # Mapping cho các giá trị đặc biệt
        special_mappings = {
            # last_new_job mapping
            'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5,
            # experience mapping  
            '<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
            '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
            '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
            '16': 16, '17': 17, '18': 18, '19': 19, '20': 20,
            '>20': 21
        }
        
        # Process data - convert các giá trị đặc biệt thành số
        data_list = []
        for line_idx, line in enumerate(data_lines):
            values = line.split(',')
            if len(values) != len(header):
                continue
                
            processed_row = []
            for i, val in enumerate(values):
                val = val.strip().replace('"', '')
                col_name = header[i] if i < len(header) else f'col_{i}'
                
                if val == '' or val.lower() in ['nan', 'null', 'none']:
                    processed_row.append(np.nan)
                else:
                    # Kiểm tra nếu là giá trị đặc biệt cần mapping
                    if val in special_mappings:
                        processed_row.append(special_mappings[val])
                    else:
                        # Thử convert sang số
                        try:
                            if '.' in val:
                                processed_row.append(float(val))
                            else:
                                processed_row.append(int(val))
                        except ValueError:
                            processed_row.append(val)  # Giữ nguyên string
            
            data_list.append(tuple(processed_row))
        
        if not data_list:
            raise ValueError("No valid data found")
        
        # Create structured array
        dtype = []
        sample_row = data_list[0]
        
        for i, val in enumerate(sample_row):
            col_name = header[i] if i < len(header) else f'col_{i}'
            if isinstance(val, (int, float)) and not np.isnan(val):
                dtype.append((col_name, np.float64))
            else:
                max_len = 1
                for row in data_list:
                    if i < len(row) and isinstance(row[i], str):
                        max_len = max(max_len, len(str(row[i])))
                dtype.append((col_name, f'U{max_len}'))
        
        data_array = np.array(data_list, dtype=dtype)
        print(f"Successfully loaded data with shape: {data_array.shape}")
        
        # Hiển thị thông tin về các cột
        print("\nColumn information:")
        for col in data_array.dtype.names:
            unique_vals = np.unique(data_array[col])
            if len(unique_vals) <= 10:  # Chỉ hiển thị nếu có ít hơn 10 giá trị unique
                print(f"  {col}: {list(unique_vals)}")
        
        return data_array
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None
def train_test_split_custom(data, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data thành train/validation/test sets chỉ dùng NumPy
    """
    np.random.seed(random_state)
    n = len(data)
    
    # Tính số lượng samples cho mỗi set
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val
    
    # Tạo indices ngẫu nhiên
    indices = np.random.permutation(n)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Split data
    train_data = data[train_indices]
    val_data = data[val_indices] 
    test_data = data[test_indices]
    
    print(f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data

def prepare_features_target_custom(data, target_col='target'):
    """
    Chuẩn bị features và target từ structured array
    """
    if target_col not in data.dtype.names:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Get feature columns (all except target và enrollee_id)
    feature_cols = [col for col in data.dtype.names 
                   if col != target_col and col != 'enrollee_id']
    
    # Create feature matrix X
    X_list = []
    feature_names = []
    
    for col in feature_cols:
        if np.issubdtype(data[col].dtype, np.number):
            X_list.append(data[col])
            feature_names.append(col)
    
    X = np.column_stack(X_list)
    y = data[target_col]
    
    print(f"Features: {X.shape[1]} numeric features")
    print(f"Feature names: {feature_names}")
    
    return X, y, feature_names
    
def validate_data_values(data):
    """
    Kiểm tra tính hợp lệ của giá trị và thống kê mô tả chỉ dùng NumPy
    """
    validation_results = {}
    
    for col_name in data.dtype.names:
        col = data[col_name]
        col_info = {}
        
        # Basic statistics for numeric columns
        if np.issubdtype(col.dtype, np.number):
            # Remove NaN for calculations
            clean_col = col[~np.isnan(col)]
            
            if len(clean_col) > 0:
                col_info['type'] = 'numeric'
                col_info['count'] = len(clean_col)
                col_info['missing'] = np.sum(np.isnan(col))
                col_info['mean'] = np.mean(clean_col)
                col_info['std'] = np.std(clean_col)
                col_info['min'] = np.min(clean_col)
                col_info['max'] = np.max(clean_col)
                col_info['median'] = np.median(clean_col)
                
                # Detect outliers using IQR (only NumPy)
                q1 = np.percentile(clean_col, 25)
                q3 = np.percentile(clean_col, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = clean_col[(clean_col < lower_bound) | (clean_col > upper_bound)]
                col_info['outliers_count'] = len(outliers)
                col_info['outliers_percentage'] = (len(outliers) / len(clean_col)) * 100
                
        else:
            # For categorical columns - chỉ dùng NumPy
            unique_vals = np.unique(col)
            counts = np.array([np.sum(col == val) for val in unique_vals])
            col_info['type'] = 'categorical'
            col_info['unique_count'] = len(unique_vals)
            col_info['most_frequent'] = unique_vals[np.argmax(counts)]
            col_info['missing'] = np.sum(col == '')
        
        validation_results[col_name] = col_info
    
    return validation_results

def detect_and_remove_outliers(data, method='iqr'):
    """
    Phát hiện và loại bỏ outliers chỉ dùng NumPy
    """
    outlier_mask = np.zeros(len(data), dtype=bool)
    
    for col_name in data.dtype.names:
        col = data[col_name]
        
        if np.issubdtype(col.dtype, np.number):
            clean_col = col[~np.isnan(col)]
            
            if len(clean_col) > 0:
                if method == 'iqr':
                    q1 = np.percentile(clean_col, 25)
                    q3 = np.percentile(clean_col, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    col_outliers = (col < lower_bound) | (col > upper_bound)
                
                elif method == 'zscore':
                    mean_val = np.mean(clean_col)
                    std_val = np.std(clean_col)
                    z_scores = np.abs((col - mean_val) / std_val)
                    col_outliers = z_scores > 3
                
                outlier_mask |= col_outliers
    
    # Remove outliers
    clean_data = data[~outlier_mask]
    print(f"Removed {np.sum(outlier_mask)} outliers from dataset")
    
    return clean_data, outlier_mask

def handle_missing_values(data, strategy='mean', fill_value=None):
    """
    Xử lý missing values với các chiến lược khác nhau chỉ dùng NumPy
    """
    processed_data = data.copy()
    
    for col_name in processed_data.dtype.names:
        col = processed_data[col_name]
        
        if np.issubdtype(col.dtype, np.number):
            nan_mask = np.isnan(col)
            
            if np.any(nan_mask):
                non_nan_vals = col[~nan_mask]
                
                if len(non_nan_vals) > 0:
                    if strategy == 'mean':
                        fill_val = np.mean(non_nan_vals)
                    elif strategy == 'median':
                        fill_val = np.median(non_nan_vals)
                    elif strategy == 'mode':
                        # Mode implementation using only NumPy
                        unique_vals, counts = np.unique(non_nan_vals, return_counts=True)
                        fill_val = unique_vals[np.argmax(counts)]
                    elif strategy == 'constant':
                        fill_val = fill_value if fill_value is not None else 0
                    else:
                        fill_val = 0
                    
                    new_col = col.copy()
                    new_col[nan_mask] = fill_val
                    processed_data[col_name] = new_col
                    
                    print(f"Filled {np.sum(nan_mask)} missing values in {col_name} with {fill_val:.4f}")
    
    return processed_data

def normalize_features(data, method='minmax'):
    """
    Chuẩn hóa features với các phương pháp khác nhau chỉ dùng NumPy
    """
    normalized_data = data.copy()
    
    for col_name in normalized_data.dtype.names:
        col = normalized_data[col_name]
        
        if np.issubdtype(col.dtype, np.number):
            clean_col = col[~np.isnan(col)]
            
            if len(clean_col) > 0:
                if method == 'minmax':
                    min_val = np.min(clean_col)
                    max_val = np.max(clean_col)
                    if max_val - min_val > 0:
                        normalized_col = (col - min_val) / (max_val - min_val)
                        normalized_data[col_name] = normalized_col
                
                elif method == 'zscore':
                    mean_val = np.mean(clean_col)
                    std_val = np.std(clean_col)
                    if std_val > 0:
                        normalized_col = (col - mean_val) / std_val
                        normalized_data[col_name] = normalized_col
                
                elif method == 'decimal':
                    max_abs = np.max(np.abs(clean_col))
                    if max_abs > 0:
                        j = int(np.ceil(np.log10(max_abs)))
                        normalized_col = col / (10 ** j)
                        normalized_data[col_name] = normalized_col
                
                elif method == 'log':
                    # Apply log transform only to positive values
                    positive_mask = col > 0
                    if np.any(positive_mask):
                        normalized_col = col.copy()
                        normalized_col[positive_mask] = np.log(normalized_col[positive_mask] + 1)
                        normalized_data[col_name] = normalized_col
    
    return normalized_data

def feature_engineering(data):
    """
    Tạo các features mới từ dữ liệu hiện có chỉ dùng NumPy
    """
    engineered_data = data.copy()
    
    # Tạo interaction features
    if 'city_development_index' in engineered_data.dtype.names and 'training_hours' in engineered_data.dtype.names:
        cdi = engineered_data['city_development_index']
        training_hrs = engineered_data['training_hours']
        
        # Interaction: development_index * training_hours
        interaction_feature = cdi * training_hrs
        engineered_data = add_column(engineered_data, 'cdi_training_interaction', interaction_feature)
        
        # Ratio: training_hours / city_development_index
        ratio_feature = np.divide(training_hrs, cdi, out=np.zeros_like(training_hrs), where=cdi != 0)
        engineered_data = add_column(engineered_data, 'training_cdi_ratio', ratio_feature)
    
    # Tạo binning features
    if 'training_hours' in engineered_data.dtype.names:
        training_hrs = engineered_data['training_hours']
        
        # Bin training hours into categories chỉ dùng NumPy
        bins = [0, 10, 50, 100, 200, np.inf]
        training_bins = np.digitize(training_hrs, bins) - 1
        training_bins = np.clip(training_bins, 0, len(bins) - 2)
        
        engineered_data = add_column(engineered_data, 'training_hours_binned', training_bins)
    
    # Tạo polynomial features
    if 'city_development_index' in engineered_data.dtype.names:
        cdi = engineered_data['city_development_index']
        engineered_data = add_column(engineered_data, 'cdi_squared', cdi ** 2)
        engineered_data = add_column(engineered_data, 'cdi_cubed', cdi ** 3)
    
    print("Feature engineering completed")
    return engineered_data

def add_column(data, col_name, new_col, dtype=np.float64):
    """
    Thêm column mới vào structured array chỉ dùng NumPy
    """
    # Create new dtype
    new_dtypes = list(data.dtype.descr)
    new_dtypes.append((col_name, dtype))
    
    # Create new array
    new_data = np.empty(len(data), dtype=new_dtypes)
    
    # Copy existing data
    for name in data.dtype.names:
        new_data[name] = data[name]
    
    # Add new column
    new_data[col_name] = new_col
    
    return new_data

def encode_categorical_features(data):
    """
    Encode categorical features using label encoding chỉ dùng NumPy
    """
    encoded_data = data.copy()
    encoders = {}
    
    for col_name in encoded_data.dtype.names:
        col = encoded_data[col_name]
        
        # Check if categorical (string type)
        if np.issubdtype(col.dtype, np.str_):
            # Get unique values chỉ dùng NumPy
            unique_vals = np.unique(col)
            # Remove empty strings and NaN equivalents
            unique_vals = [val for val in unique_vals if val not in ['', 'nan', 'null']]
            
            if unique_vals:
                encoder_dict = {val: idx for idx, val in enumerate(unique_vals)}
                encoders[col_name] = encoder_dict
                
                # Encode column chỉ dùng NumPy
                encoded_col = np.array([encoder_dict.get(val, -1) for val in col], dtype=np.float64)
                encoded_data = _replace_column(encoded_data, col_name, encoded_col, np.float64)
    
    return encoded_data, encoders

def _replace_column(data, col_name, new_col, new_dtype):
    """
    Thay thế column trong structured array chỉ dùng NumPy
    """
    new_dtypes = []
    for name in data.dtype.names:
        if name == col_name:
            new_dtypes.append((name, new_dtype))
        else:
            new_dtypes.append((name, data.dtype[name]))
    
    new_data = np.empty(len(data), dtype=new_dtypes)
    
    for name in data.dtype.names:
        if name == col_name:
            new_data[name] = new_col
        else:
            new_data[name] = data[name]
    
    return new_data

def prepare_features_target(data, target_col='target'):
    """
    Chuẩn bị features và target cho modeling chỉ dùng NumPy
    """
    if target_col not in data.dtype.names:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Get feature columns (all except target)
    feature_cols = [col for col in data.dtype.names if col != target_col]
    
    # Create feature matrix X chỉ dùng NumPy
    X_list = []
    for col in feature_cols:
        if np.issubdtype(data[col].dtype, np.number):
            X_list.append(data[col])
    
    X = np.column_stack(X_list)
    y = data[target_col]
    
    return X, y

def compute_descriptive_statistics(data):
    """
    Tính toán thống kê mô tả chỉ dùng NumPy
    """
    stats = {}
    
    for col_name in data.dtype.names:
        col = data[col_name]
        col_stats = {}
        
        if np.issubdtype(col.dtype, np.number):
            clean_col = col[~np.isnan(col)]
            
            if len(clean_col) > 0:
                col_stats['count'] = len(clean_col)
                col_stats['mean'] = np.mean(clean_col)
                col_stats['std'] = np.std(clean_col)
                col_stats['variance'] = np.var(clean_col)
                col_stats['min'] = np.min(clean_col)
                col_stats['max'] = np.max(clean_col)
                col_stats['range'] = np.max(clean_col) - np.min(clean_col)
                col_stats['median'] = np.median(clean_col)
                col_stats['q1'] = np.percentile(clean_col, 25)
                col_stats['q3'] = np.percentile(clean_col, 75)
                col_stats['iqr'] = np.percentile(clean_col, 75) - np.percentile(clean_col, 25)
                col_stats['skewness'] = _compute_skewness(clean_col)  # Custom implementation
                
        stats[col_name] = col_stats
    
    return stats

def _compute_skewness(data):
    """
    Tính skewness chỉ dùng NumPy
    """
    n = len(data)
    if n < 3:
        return 0
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0
    
    # Calculate skewness
    skew = (np.sum((data - mean) ** 3) / n) / (std ** 3)
    return skew

def statistical_hypothesis_test_numpy(data, col1, col2, alpha=0.05):
    """
    Kiểm định giả thuyết thống kê chỉ dùng NumPy
    H0: Không có sự khác biệt giữa hai groups
    H1: Có sự khác biệt giữa hai groups
    """
    if col1 not in data.dtype.names or col2 not in data.dtype.names:
        raise ValueError("Column names not found in data")
    
    group1 = data[col1]
    group2 = data[col2]
    
    # Remove NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    # Two-sample t-test implementation using only NumPy
    n1, n2 = len(group1_clean), len(group2_clean)
    mean1, mean2 = np.mean(group1_clean), np.mean(group2_clean)
    var1, var2 = np.var(group1_clean, ddof=1), np.var(group2_clean, ddof=1)
    
    # Pooled standard error
    pooled_se = np.sqrt((var1 / n1) + (var2 / n2))
    
    # T-statistic
    t_stat = (mean1 - mean2) / pooled_se
    
    # Degrees of freedom (approximate - Welch's t-test)
    df_numerator = (var1 / n1 + var2 / n2) ** 2
    df_denominator = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = df_numerator / df_denominator
    
    # Manual p-value calculation using t-distribution approximation
    # This is a simplified version - for exact values you'd need scipy
    p_value = _manual_t_test_p_value(t_stat, df)
    
    result = {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean1 - mean2,
        'reject_null': p_value < alpha,
        'hypothesis': {
            'H0': f'Mean({col1}) = Mean({col2})',
            'H1': f'Mean({col1}) ≠ Mean({col2})'
        }
    }
    
    return result

def _manual_t_test_p_value(t_stat, df):
    """
    Tính p-value sử dụng incomplete beta function
    Công thức chính xác: p = 2 * (1 - I_{df/(df+t²)}(df/2, 1/2))
    """
    t = abs(t_stat)
    x = df / (df + t**2)
    
    # Tính regularized incomplete beta function I_x(a,b)
    def beta_inc(a, b, x, max_iter=1000):
        """Tính incomplete beta function I_x(a, b)"""
        if x == 0:
            return 0
        if x == 1:
            return 1
        
        # Sử dụng continued fraction (công thức chính xác)
        az = 1.0
        bz = 1.0 - (a + b) * x / (a + 1.0)
        am = 1.0
        bm = 1.0
        
        for i in range(1, max_iter + 1):
            em = float(i)
            tem = em + em
            d = em * (b - em) * x / ((a + tem - 1) * (a + tem))
            ap = az + d * am
            bp = bz + d * bm
            d = -(a + em) * (a + b + em) * x / ((a + tem) * (a + tem + 1))
            app = ap + d * az
            bpp = bp + d * bz
            
            if abs(app - az) < 1e-10 * abs(az):
                break
                
            am = ap / bpp
            bm = bp / bpp
            az = app / bpp
            bz = 1.0
        
        return az
    
    a = df / 2.0
    b = 0.5
    beta_val = beta_inc(a, b, x)
    
    p_value = beta_val  # Đây là one-tailed p-value
    p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed
    
    return p_value

def handle_missing(data):
    """Handle missing: mode for cat, mean for num."""
    fields = data.dtype.names
    for field in fields:
        col = data[field]
        if np.issubdtype(col.dtype, np.number):
            mask_missing = np.isnan(col)
            if np.any(mask_missing):
                mean_val = np.nanmean(col)
                col[mask_missing] = mean_val
        else:
            mask_missing = (col == '') | (col == 'nan') | (col == 'NaN')
            if np.any(mask_missing):
                non_missing = col[~mask_missing]
                unique, counts = np.unique(non_missing, return_counts=True)
                mode = unique[np.argmax(counts)]
                col[mask_missing] = mode
    return data

def encode_and_engineer_features(data):
    """Encode cat to num, engineering."""
    # Numerical
    city_dev_index = data['city_development_index'].astype(float)
    training_hours = data['training_hours'].astype(float)
    target = data['target'].astype(float) if 'target' in data.dtype.names else None

    # Categorical mappings (full unique)
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    gender_encoded = np.array([gender_map.get(g, 0) for g in data['gender']])

    relevent_exp_map = {'Has relevent experience': 1, 'No relevent experience': 0}
    relevent_exp_encoded = np.array([relevent_exp_map.get(e, 0) for e in data['relevent_experience']])

    enrolled_uni_map = {'no_enrollment': 0, 'Part time course': 1, 'Full time course': 2}
    enrolled_uni_encoded = np.array([enrolled_uni_map.get(u, 0) for u in data['enrolled_university']])

    edu_level_map = {'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4}
    edu_level_encoded = np.array([edu_level_map.get(l, 2) for l in data['education_level']])

    # major_discipline: One-hot (6)
    major_unique = ['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other']
    major_map = {m: i for i, m in enumerate(major_unique)}
    major_encoded = np.zeros((len(data), len(major_unique)))
    for i, m in enumerate(data['major_discipline']):
        if m in major_map:
            major_encoded[i, major_map[m]] = 1
        else:
            major_encoded[i, 4] = 1  # Default No Major

    # experience: Full <1 to >20
    exp_map = {'<1': 0, '>20': 21}
    for i in range(1, 21):
        exp_map[f'{i}'] = i
    exp_encoded = np.array([exp_map.get(e, 10) for e in data['experience']])

    # company_size: Full, numerical approx
    comp_size_map = {'<10': 5, '10-49': 30, '50-99': 75, '100-500': 300, '500-999': 750,
                     '1000-4999': 3000, '5000-9999': 7500, '10000+': 15000}
    comp_size_encoded = np.array([comp_size_map.get(s, 300) for s in data['company_size']])

    # company_type: One-hot (6)
    comp_type_unique = ['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Other', 'Public Sector', 'NGO']
    comp_type_map = {t: i for i, t in enumerate(comp_type_unique)}
    comp_type_encoded = np.zeros((len(data), len(comp_type_unique)))
    for i, t in enumerate(data['company_type']):
        if t in comp_type_map:
            comp_type_encoded[i, comp_type_map[t]] = 1
        else:
            comp_type_encoded[i, 3] = 1  # Default Other

    # last_new_job: Full
    last_job_map = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}
    last_job_encoded = np.array([last_job_map.get(j, 1) for j in data['last_new_job']])

    # Stack X
    X = np.column_stack((
        city_dev_index, training_hours, gender_encoded, relevent_exp_encoded,
        enrolled_uni_encoded, edu_level_encoded, exp_encoded, comp_size_encoded,
        last_job_encoded, major_encoded, comp_type_encoded
    ))

    # Engineering: exp * edu
    interaction = exp_encoded * edu_level_encoded
    X = np.column_stack((X, interaction))

    return X, target

def normalize_minmax(X):
    """Min-max normalization chỉ dùng NumPy"""
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)