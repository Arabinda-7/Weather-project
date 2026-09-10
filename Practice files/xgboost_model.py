"""
=============================================================================
XGBoost Weather Prediction Model - Training, Testing & Comprehensive Evaluation
Dataset: bengaluru_preprocessed.csv
Target: 2-meter Air Temperature (temperature_2m) / Weather Forecasting
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score
)
from sklearn.preprocessing import StandardScaler

# Configure warnings and aesthetics
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# 1. DATA LOADING & AUDIT
# =============================================================================
def load_data(file_path='bengaluru_preprocessed.csv'):
    """Load the preprocessed dataset and display basic diagnostics."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset '{file_path}' not found. Please ensure it is in the working directory.")
    
    print("=" * 70)
    print(" STEP 1: LOADING DATASET")
    print("=" * 70)
    df = pd.read_csv(file_path)
    print(f"Loaded dataset successfully from: {file_path}")
    print(f"Initial Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 2. FEATURE ENGINEERING & MISSING PREPROCESSING STEPS
# =============================================================================
def preprocess_and_engineer_features(df, target_col='temperature_2m'):
    """
    Addresses what was missed in previous preprocessing:
    1. Parses datetime correctly and sorts chronologically.
    2. Encodes cyclical temporal features (sine/cosine for hour & month).
    3. Generates lag and rolling window features for temporal momentum.
    4. One-Hot encodes remaining categorical columns (season, day_name).
    5. Cleans and structures feature matrices without lookahead data leakage.
    """
    print("\n" + "=" * 70)
    print(" STEP 2: ADVANCED FEATURE ENGINEERING & PREPROCESSING AUDIT")
    print("=" * 70)
    
    data = df.copy()
    
    # 2.1 Datetime parsing and chronological sorting
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time').reset_index(drop=True)
    print(f"Time series span: {data['time'].min()} to {data['time'].max()}")
    
    # 2.2 Cyclical Temporal Features (Sine & Cosine transformations)
    # Hour (24 hours period)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24.0)
    # Month (12 months period)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)
    # Day of Year (365.25 days period)
    data['day_of_year'] = data['time'].dt.dayofyear
    data['doy_sin'] = np.sin(2 * np.pi * data['day_of_year'] / 365.25)
    data['doy_cos'] = np.cos(2 * np.pi * data['day_of_year'] / 365.25)
    print("Added cyclical trigonometric features: hour_sin/cos, month_sin/cos, doy_sin/cos")
    
    # 2.3 Lag features for key meteorological variables (captures temporal inertia)
    lag_cols = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m']
    lag_cols = [c for c in lag_cols if c in data.columns]
    
    for col in lag_cols:
        data[f'{col}_lag_1h'] = data[col].shift(1)
        data[f'{col}_lag_24h'] = data[col].shift(24)
        
    # 2.4 Rolling statistical aggregations (past 24h rolling mean and std)
    data['temp_rolling_mean_24h'] = data['temperature_2m'].shift(1).rolling(window=24).mean()
    data['temp_rolling_std_24h'] = data['temperature_2m'].shift(1).rolling(window=24).std()
    data['humidity_rolling_mean_24h'] = data['relative_humidity_2m'].shift(1).rolling(window=24).mean()
    
    print("Added temporal lag features (1h, 24h) and 24-hour rolling statistics")
    
    # Drop rows with NaN resulting from lag / rolling operations
    initial_rows = len(data)
    data = data.dropna().reset_index(drop=True)
    print(f"Dropped {initial_rows - len(data)} rows with warm-up NaNs from lag features.")
    
    # 2.5 Categorical One-Hot Encoding
    categorical_cols = [c for c in ['season', 'day_name'] if c in data.columns]
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=float)
        print(f"One-hot encoded categorical variables: {categorical_cols}")
        
    return data


# =============================================================================
# 3. TIME-SERIES TRAIN-VALIDATION-TEST SPLIT
# =============================================================================
def split_data(data, target_col='temperature_2m', train_ratio=0.8, val_ratio=0.1):
    """
    Performs a strict chronological split (Time-Series aware) to prevent
    lookahead data leakage.
    """
    print("\n" + "=" * 70)
    print(" STEP 3: TIME-SERIES DATA SPLITTING (CHRONOLOGICAL)")
    print("=" * 70)
    
    # Drop non-feature columns
    drop_cols = ['time', target_col]
    
    # Prevent direct target leakage if highly collinear same-hour derivatives exist
    leaking_cols = ['apparent_temperature', 'dew_point_2m', 'soil_temperature_0_to_7cm']
    drop_cols.extend([c for c in leaking_cols if c in data.columns and c != target_col])
    
    feature_cols = [c for c in data.columns if c not in drop_cols]
    
    X = data[feature_cols]
    y = data[target_col]
    timestamps = data['time']
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    time_train = (timestamps.iloc[0], timestamps.iloc[train_end - 1])
    time_val = (timestamps.iloc[train_end], timestamps.iloc[val_end - 1])
    time_test = (timestamps.iloc[val_end], timestamps.iloc[-1])
    
    print(f"Total Samples     : {n:,}")
    print(f"Training Set (80%): {len(X_train):,} samples | Period: {time_train[0].date()} -> {time_train[1].date()}")
    print(f"Validation Set(10%): {len(X_val):,} samples | Period: {time_val[0].date()} -> {time_val[1].date()}")
    print(f"Test Set (10%)    : {len(X_test):,} samples | Period: {time_test[0].date()} -> {time_test[1].date()}")
    print(f"Number of Features: {X.shape[1]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, timestamps.iloc[val_end:].values, feature_cols


# =============================================================================
# 4. MODEL INITIALIZATION & TRAINING
# =============================================================================
def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Initializes and trains the XGBoost Regressor with early stopping."""
    print("\n" + "=" * 70)
    print(" STEP 4: TRAINING XGBOOST REGRESSOR")
    print("=" * 70)
    
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.03,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'early_stopping_rounds': 50
    }
    
    print("Model Hyperparameters:")
    for k, v in params.items():
        print(f"  • {k:<22}: {v}")
    
    model = xgb.XGBRegressor(**params)
    
    print("\nTraining in progress...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    best_iteration = model.best_iteration
    best_score = model.best_score
    print(f"\nTraining Complete!")
    print(f"Best Iteration: {best_iteration} | Validation RMSE: {best_score:.4f}")
    
    return model


# =============================================================================
# 5. MODEL EVALUATION & PERFORMANCE METRICS
# =============================================================================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Calculates and prints comprehensive evaluation metrics on train and test sets."""
    print("\n" + "=" * 70)
    print(" STEP 5: COMPREHENSIVE PERFORMANCE EVALUATION")
    print("=" * 70)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    def calculate_metrics(y_true, y_pred, split_name="Test"):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        evs = explained_variance_score(y_true, y_pred)
        
        return {
            'Split': split_name,
            'RMSE (°C)': rmse,
            'MAE (°C)': mae,
            'MSE (°C²)': mse,
            'R² Score': r2,
            'MAPE (%)': mape,
            'Explained Variance': evs
        }
    
    metrics_train = calculate_metrics(y_train, y_pred_train, "Train")
    metrics_test = calculate_metrics(y_test, y_pred_test, "Test (Unseen)")
    
    metrics_df = pd.DataFrame([metrics_train, metrics_test]).set_index('Split')
    print("\nPerformance Comparison Table:")
    print(metrics_df.to_string())
    
    return y_pred_test, metrics_df


# =============================================================================
# 6. VISUALIZATION & DIAGNOSTICS
# =============================================================================
def generate_evaluation_plots(model, y_test, y_pred_test, test_timestamps, feature_names, output_dir='.'):
    """Generates evaluation plots: Loss Curves, Actual vs Pred, Residuals, and Feature Importance."""
    print("\n" + "=" * 70)
    print(" STEP 6: GENERATING EVALUATION DIAGNOSTICS & PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Learning Curves (Training vs Validation RMSE)
    evals_result = model.evals_result()
    train_rmse = evals_result['validation_0']['rmse']
    val_rmse = evals_result['validation_1']['rmse']
    epochs = range(len(train_rmse))
    
    axes[0, 0].plot(epochs, train_rmse, label='Train RMSE', color='#1f77b4', lw=2)
    axes[0, 0].plot(epochs, val_rmse, label='Validation RMSE', color='#ff7f0e', lw=2)
    axes[0, 0].axvline(model.best_iteration, color='red', linestyle='--', alpha=0.7, label=f'Best Iter ({model.best_iteration})')
    axes[0, 0].set_title('Training & Validation RMSE Learning Curve', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Boosting Iterations')
    axes[0, 0].set_ylabel('RMSE (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted (Time-Series Zoom - Last 300 Hours)
    zoom_len = min(300, len(y_test))
    axes[0, 1].plot(range(zoom_len), y_test.iloc[-zoom_len:].values, label='Actual Temperature', color='#2ca02c', lw=2)
    axes[0, 1].plot(range(zoom_len), y_pred_test[-zoom_len:], label='XGBoost Predicted', color='#d62728', linestyle='--', lw=2)
    axes[0, 1].set_title(f'Actual vs. Predicted Temperature (Recent {zoom_len} Hours Zoom)', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Recent Hourly Time Steps')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution & Q-Q Summary
    residuals = y_test - y_pred_test
    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#9467bd', bins=40)
    axes[1, 0].axvline(0, color='black', linestyle='--', lw=1.5)
    axes[1, 0].set_title(f'Residuals Distribution (Mean={residuals.mean():.3f}, Std={residuals.std():.3f})', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Prediction Error (Actual - Predicted) °C')
    axes[1, 0].set_ylabel('Density / Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Top 15 Feature Importances (Gain)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Top 15 Feature Importance (Gain)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Relative Importance Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'xgboost_evaluation_plots.png')
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    print(f"Saved evaluation plots to: {plot_file}")


# =============================================================================
# 7. MODEL PERSISTENCE & INFERENCE DEMO
# =============================================================================
def save_model_and_artifacts(model, feature_names, metrics_df, output_dir='.'):
    """Saves the trained model, feature metadata, and performance summary."""
    print("\n" + "=" * 70)
    print(" STEP 7: SAVING MODEL & METADATA ARTIFACTS")
    print("=" * 70)
    
    model_joblib_path = os.path.join(output_dir, 'xgboost_weather_model.joblib')
    model_json_path = os.path.join(output_dir, 'xgboost_weather_model.json')
    meta_path = os.path.join(output_dir, 'xgboost_model_metadata.json')
    
    # Save joblib and native JSON format
    joblib.dump(model, model_joblib_path)
    model.save_model(model_json_path)
    
    metadata = {
        'model_name': 'XGBoost Regressor (Weather Forecast)',
        'target_variable': 'temperature_2m',
        'n_features': len(feature_names),
        'features': feature_names,
        'test_metrics': metrics_df.loc['Test (Unseen)'].to_dict()
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Saved model (joblib)  : {model_joblib_path}")
    print(f"Saved model (JSON)    : {model_json_path}")
    print(f"Saved metadata (JSON) : {meta_path}")


def sample_inference_demo(model, X_test, y_test):
    """Demonstrates how to perform inference on new unseen samples."""
    print("\n" + "=" * 70)
    print(" STEP 8: SAMPLE INFERENCE DEMONSTRATION")
    print("=" * 70)
    
    sample_X = X_test.iloc[:5]
    sample_actual = y_test.iloc[:5].values
    sample_preds = model.predict(sample_X)
    
    demo_df = pd.DataFrame({
        'Actual Temp (°C)': np.round(sample_actual, 2),
        'Predicted Temp (°C)': np.round(sample_preds, 2),
        'Absolute Error (°C)': np.round(np.abs(sample_actual - sample_preds), 2)
    })
    print("Inference on 5 unseen test samples:")
    print(demo_df.to_string(index=False))


# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    print("#####################################################################")
    print("#      XGBOOST WEATHER FORECASTING & EVALUATION PIPELINE            #")
    print("#####################################################################")
    
    # 1. Load data
    df = load_data('bengaluru_preprocessed.csv')
    
    # 2. Feature Engineering & Preprocessing Audit
    data = preprocess_and_engineer_features(df, target_col='temperature_2m')
    
    # 3. Train-Val-Test Chronological Split
    X_train, y_train, X_val, y_val, X_test, y_test, test_timestamps, feature_names = split_data(data)
    
    # 4. Train Model
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate Model
    y_pred_test, metrics_df = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # 6. Generate Plots
    generate_evaluation_plots(model, y_test, y_pred_test, test_timestamps, feature_names)
    
    # 7. Save Model Artifacts
    save_model_and_artifacts(model, feature_names, metrics_df)
    
    # 8. Sample Inference
    sample_inference_demo(model, X_test, y_test)
    
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
