import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def setup_plot_style():
    """Setup consistent plot style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_target_distribution(data):
    """Plot distribution of target variable - Histogram + Pie chart"""
    setup_plot_style()
    
    if 'target' not in data.dtype.names:
        print("Target column not found")
        return
    
    target = data['target']
    unique_vals, counts = np.unique(target, return_counts=True)
    total = len(target)
    
    plt.figure(figsize=(15, 5))
    
    # Bar plot - Histogram
    plt.subplot(1, 3, 1)
    bars = plt.bar([str(x) for x in unique_vals], counts, 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
    plt.title('Phân phối biến mục tiêu (Histogram)')
    plt.xlabel('Giá trị Target')
    plt.ylabel('Số lượng')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', va='bottom')
    
    # Pie chart - Tỷ lệ phần trăm
    plt.subplot(1, 3, 2)
    labels = [f'Không đổi việc\n({counts[0]})', f'Đổi việc\n({counts[1]})']
    percentages = [counts[0]/total*100, counts[1]/total*100]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['lightblue', 'lightcoral'])
    plt.title('Tỷ lệ phần trăm biến mục tiêu')
    
    # Statistical summary
    plt.subplot(1, 3, 3)
    stats_text = f"""
    THỐNG KÊ BIẾN MỤC TIÊU:
    
    Tổng số mẫu: {total}
    Số không đổi việc: {counts[0]}
    Số đổi việc: {counts[1]}
    Tỷ lệ đổi việc: {counts[1]/total*100:.2f}%
    
    Imbalance Ratio: {counts[0]/counts[1]:.2f}:1
    """
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_numeric_feature_distribution(data, column_name):
    """Plot distribution of numeric feature - Histogram + Box plot"""
    setup_plot_style()
    
    if column_name not in data.dtype.names:
        print(f"Column {column_name} not found")
        return
    
    col_data = data[column_name]
    
    if not np.issubdtype(col_data.dtype, np.number):
        print(f"Column {column_name} is not numeric")
        return
    
    clean_data = col_data[~np.isnan(col_data)]
    
    plt.figure(figsize=(15, 5))
    
    # Histogram
    plt.subplot(1, 3, 1)
    plt.hist(clean_data, bins=30, alpha=0.7, color='lightseagreen', edgecolor='black')
    plt.title(f'Phân phối {column_name} (Histogram)')
    plt.xlabel(column_name)
    plt.ylabel('Tần suất')
    
    # Box plot
    plt.subplot(1, 3, 2)
    plt.boxplot(clean_data)
    plt.title(f'Phân phối {column_name} (Box Plot)')
    plt.ylabel(column_name)
    
    # Statistical summary
    plt.subplot(1, 3, 3)
    stats_text = f"""
    THỐNG KÊ {column_name}:
    
    Count: {len(clean_data)}
    Mean: {np.mean(clean_data):.2f}
    Std: {np.std(clean_data):.2f}
    Min: {np.min(clean_data):.2f}
    Max: {np.max(clean_data):.2f}
    Median: {np.median(clean_data):.2f}
    Q1: {np.percentile(clean_data, 25):.2f}
    Q3: {np.percentile(clean_data, 75):.2f}
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_categorical_feature_distribution(data, column_name):
    """Plot distribution of categorical feature - Bar chart + Pie chart"""
    setup_plot_style()
    
    if column_name not in data.dtype.names:
        print(f"Column {column_name} not found")
        return
    
    col_data = data[column_name]
    unique_vals, counts = np.unique(col_data, return_counts=True)
    total = len(col_data)
    
    plt.figure(figsize=(15, 5))
    
    # Bar chart
    plt.subplot(1, 3, 1)
    bars = plt.bar(range(len(unique_vals)), counts, color='lightsteelblue', alpha=0.7)
    plt.title(f'Phân phối {column_name} (Bar Chart)')
    plt.xlabel(column_name)
    plt.ylabel('Số lượng')
    plt.xticks(range(len(unique_vals)), unique_vals, rotation=45)
    
    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    plt.subplot(1, 3, 2)
    labels = [f'{val}\n({count})' for val, count in zip(unique_vals, counts)]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Phân phối {column_name} (Pie Chart)')
    
    # Statistical summary
    plt.subplot(1, 3, 3)
    stats_text = f"""
    THỐNG KÊ {column_name}:
    
    Tổng số: {total}
    Số loại: {len(unique_vals)}
    Phổ biến nhất: {unique_vals[np.argmax(counts)]}
    Tỷ lệ phổ biến: {np.max(counts)/total*100:.1f}%
    """
    plt.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_scatter_correlation(data, x_col, y_col):
    """Scatter plot to show correlation between two numeric variables"""
    setup_plot_style()
    
    if x_col not in data.dtype.names or y_col not in data.dtype.names:
        print("Columns not found")
        return
    
    x_data = data[x_col]
    y_data = data[y_col]
    
    if not (np.issubdtype(x_data.dtype, np.number) and 
            np.issubdtype(y_data.dtype, np.number)):
        print("Both columns must be numeric")
        return
    
    # Remove NaN values
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[mask]
    y_clean = y_data[mask]
    
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(x_clean, y_clean, alpha=0.6, color='steelblue')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Scatter Plot: {x_col} vs {y_col}')
    
    # Add trend line
    if len(x_clean) > 1:
        # Linear regression using NumPy
        A = np.vstack([x_clean, np.ones(len(x_clean))]).T
        m, c = np.linalg.lstsq(A, y_clean, rcond=None)[0]
        plt.plot(x_clean, m*x_clean + c, 'r-', linewidth=2, 
                label=f'y = {m:.2f}x + {c:.2f}')
        plt.legend()
    
    # Correlation heatmap
    plt.subplot(1, 2, 2)
    corr_matrix = np.corrcoef([x_clean, y_clean])
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=[x_col, y_col], yticklabels=[x_col, y_col])
    plt.title('Ma trận tương quan')
    
    plt.tight_layout()
    plt.show()
    
    # Print correlation coefficient
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    print(f"Hệ số tương quan giữa {x_col} và {y_col}: {correlation:.3f}")

def plot_feature_vs_target(data, feature_col, target_col='target'):
    """Plot relationship between feature and target variable"""
    setup_plot_style()
    
    if feature_col not in data.dtype.names or target_col not in data.dtype.names:
        print("Columns not found")
        return
    
    feature = data[feature_col]
    target = data[target_col]
    
    plt.figure(figsize=(15, 5))
    
    if np.issubdtype(feature.dtype, np.number):
        # Numeric feature
        plt.subplot(1, 3, 1)
        # Distribution by target
        for target_val in np.unique(target):
            mask = target == target_val
            plt.hist(feature[mask], alpha=0.7, label=f'Target={target_val}', 
                    bins=20, density=True)
        plt.xlabel(feature_col)
        plt.ylabel('Mật độ')
        plt.legend()
        plt.title(f'Phân phối {feature_col} theo Target')
        
        plt.subplot(1, 3, 2)
        # Box plot by target
        plot_data = [feature[target == val] for val in np.unique(target)]
        plt.boxplot(plot_data, labels=[f'Target={val}' for val in np.unique(target)])
        plt.ylabel(feature_col)
        plt.title(f'Box Plot {feature_col} theo Target')
        
        plt.subplot(1, 3, 3)
        # Mean comparison
        means = [np.mean(feature[target == val]) for val in np.unique(target)]
        stds = [np.std(feature[target == val]) for val in np.unique(target)]
        
        bars = plt.bar(range(len(means)), means, yerr=stds, capsize=5,
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        plt.xticks(range(len(means)), [f'Target={val}' for val in np.unique(target)])
        plt.ylabel(f'Mean {feature_col}')
        plt.title('So sánh giá trị trung bình')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean:.2f}', ha='center', va='bottom')
    
    else:
        # Categorical feature
        unique_vals = np.unique(feature)
        change_rates = []
        
        for val in unique_vals:
            mask = feature == val
            total_in_group = np.sum(mask)
            if total_in_group > 0:
                change_rate = np.sum(target[mask] == 1) / total_in_group * 100
                change_rates.append(change_rate)
            else:
                change_rates.append(0)
        
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(unique_vals)), change_rates, color='mediumpurple', alpha=0.7)
        plt.xticks(range(len(unique_vals)), unique_vals, rotation=45)
        plt.xlabel(feature_col)
        plt.ylabel('Tỷ lệ đổi việc (%)')
        plt.title(f'Tỷ lệ đổi việc theo {feature_col}')
        
        # Add value labels
        for bar, rate in zip(bars, change_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(data, numeric_columns=None):
    """Plot correlation heatmap for numeric columns"""
    setup_plot_style()
    
    if numeric_columns is None:
        # Auto-detect numeric columns
        numeric_columns = []
        for col in data.dtype.names:
            if np.issubdtype(data[col].dtype, np.number) and col != 'enrollee_id':
                numeric_columns.append(col)
    
    if len(numeric_columns) < 2:
        print("Need at least 2 numeric columns for correlation heatmap")
        return
    
    # Create correlation matrix using only NumPy
    numeric_data = []
    valid_columns = []
    
    for col in numeric_columns:
        col_data = data[col]
        clean_data = col_data[~np.isnan(col_data)]
        if len(clean_data) > 1:  # Need at least 2 points for correlation
            numeric_data.append(col_data)
            valid_columns.append(col)
    
    if len(numeric_data) < 2:
        print("Not enough valid numeric data for correlation")
        return
    
    # Handle missing values by taking available pairs
    corr_matrix = np.ones((len(valid_columns), len(valid_columns)))
    
    for i in range(len(valid_columns)):
        for j in range(i+1, len(valid_columns)):
            # Only use pairs where both values are not NaN
            mask = ~(np.isnan(numeric_data[i]) | np.isnan(numeric_data[j]))
            if np.sum(mask) > 1:
                corr = np.corrcoef(numeric_data[i][mask], numeric_data[j][mask])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                xticklabels=valid_columns, yticklabels=valid_columns,
                square=True, fmt='.2f')
    plt.title('Ma trận tương quan giữa các biến số', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_multiple_features(data, columns, ncols=3):
    """Plot multiple features in a grid"""
    setup_plot_style()
    
    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes
    
    for idx, col in enumerate(columns):
        if col not in data.dtype.names:
            continue
            
        ax = axes[idx]
        col_data = data[col]
        
        if np.issubdtype(col_data.dtype, np.number):
            # Numeric - histogram
            clean_data = col_data[~np.isnan(col_data)]
            ax.hist(clean_data, bins=20, alpha=0.7, color='lightseagreen', edgecolor='black')
            ax.set_title(f'{col}')
            ax.set_xlabel('Giá trị')
            ax.set_ylabel('Tần suất')
        else:
            # Categorical - bar chart
            unique_vals, counts = np.unique(col_data, return_counts=True)
            # Show top 10 categories if too many
            if len(unique_vals) > 10:
                sorted_idx = np.argsort(counts)[-10:]
                unique_vals = unique_vals[sorted_idx]
                counts = counts[sorted_idx]
            
            bars = ax.bar(range(len(unique_vals)), counts, color='steelblue', alpha=0.7)
            ax.set_title(f'{col}')
            ax.set_xlabel('Categories')
            ax.set_ylabel('Số lượng')
            ax.set_xticks(range(len(unique_vals)))
            ax.set_xticklabels(unique_vals, rotation=45, ha='right')
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()