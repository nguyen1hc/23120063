from .data_processing import (
    load_data,
    handle_missing,
    encode_and_engineer_features,
    normalize_minmax,
    validate_data_values,
    detect_and_remove_outliers,
    handle_missing_values,
    normalize_features,
    feature_engineering,
    encode_categorical_features,
    prepare_features_target,
    compute_descriptive_statistics,
    statistical_hypothesis_test_numpy,
    compute_descriptive_statistics,
    statistical_hypothesis_test_numpy,
    detect_and_remove_outliers,
)

from .visualization import (
    plot_target_distribution,
    plot_numeric_feature_distribution,
    plot_categorical_feature_distribution,
    plot_scatter_correlation,
    plot_feature_vs_target,
    plot_correlation_heatmap,
    plot_multiple_features,
    setup_plot_style
)

from .models import (
    LinearRegression, LogisticRegression, KNN,
    EvaluationMetrics, CrossValidation, StandardScaler,
)