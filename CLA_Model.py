import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, LSTM, MaxPooling1D, Dropout
from scikeras.wrappers import KerasRegressor
from scipy.optimize import minimize
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import joblib
import pickle


def LCCC(y_true, y_pred):
    y_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    covariance = np.mean((y_true - y_mean) * (y_pred - y_pred_mean))
    y_std = np.std(y_true, ddof=1)
    y_pred_std = np.std(y_pred, ddof=1)
    lccc = (2 * covariance) / (y_std ** 2 + y_pred_std ** 2 + (y_mean - y_pred_mean) ** 2)
    return lccc

def evaluate_model(model, X, y, scaler_y):
    y_pred_scaled = model.predict(X)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    lc = LCCC(y, y_pred)
    return r2, rmse, lc, y_pred

data = np.array(pd.read_csv('DATA.csv'))

x, y = data[:, 1:], data[:, 0]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(x)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.33, random_state=42)

input_shape_cnn = (X_train.shape[1], 1)
input_shape_lstm = (X_train.shape[1], 1)
input_shape_ann = X_train.shape[1]

def create_cnn(input_shape):     #  relu tanh linear
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='tanh', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.8))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model

def create_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(150, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.6))
    model.add(Dense(150, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='tanh'))
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model

def create_ann(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='tanh'))
    model.add(Dense(128, activation='tanh', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model

cnn_model = KerasRegressor(model=create_cnn, input_shape=input_shape_cnn, epochs=250, batch_size=32, verbose=1)
lstm_model = KerasRegressor(model=create_lstm, input_shape=input_shape_lstm, epochs=500, batch_size=16, verbose=1)
ann_model = KerasRegressor(model=create_ann, input_shape=input_shape_ann, epochs=250, batch_size=16, verbose=1)

models = [cnn_model, lstm_model, ann_model]
model_names = ['CNN', 'LSTM', 'ANN']

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((X_train.shape[0], len(models)))

all_metrics = {name: {'train': {'R2': [], 'RMSE': [], 'LCCC': []},
                      'val': {'R2': [], 'RMSE': [], 'LCCC': []}} for name in model_names}

for i, model in enumerate(models):
    model_name = model_names[i]
    print(f"Evaluating {model_name}...")
    for train_idx, val_idx in kfold.split(X_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        # Train metrics
        train_r2, train_rmse, train_lc, train_pred = evaluate_model(model, X_train[train_idx], y_train[train_idx], scaler_y)
        all_metrics[model_name]['train']['R2'].append(train_r2)
        all_metrics[model_name]['train']['RMSE'].append(train_rmse)
        all_metrics[model_name]['train']['LCCC'].append(train_lc)
        # Validation metrics
        val_r2, val_rmse, val_lc, val_pred = evaluate_model(model, X_train[val_idx], y_train[val_idx], scaler_y)
        all_metrics[model_name]['val']['R2'].append(val_r2)
        all_metrics[model_name]['val']['RMSE'].append(val_rmse)
        all_metrics[model_name]['val']['LCCC'].append(val_lc)
        meta_features[val_idx, i] = val_pred

def loss_function(weights):
    weights = np.array(weights)
    weighted_preds = np.dot(meta_features, weights)
    r2 = r2_score(y_train, weighted_preds)
    return -r2

initial_weights = np.ones(len(models)) / len(models)

constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)},
               {'type': 'ineq', 'fun': lambda w: w})

optimized_weights = minimize(loss_function, initial_weights, constraints=constraints).x

for model in models:
    model.fit(X_train, y_train)

def super_learner_predict(X):
    predictions = np.column_stack([model.predict(X) for model in models])
    return np.dot(predictions, optimized_weights)

# 反归一化过程
def reverse_scaling(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

# 对训练和测试数据进行预测
y_train_pred_scaled = super_learner_predict(X_train)
y_test_pred_scaled = super_learner_predict(X_test)

# 反归一化处理预测结果
y_train_pred = reverse_scaling(y_train_pred_scaled, scaler_y)
y_test_pred = reverse_scaling(y_test_pred_scaled, scaler_y)

# 反归一化处理实际值
y_train = reverse_scaling(y_train, scaler_y)
y_test = reverse_scaling(y_test, scaler_y)

# 计算指标
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_lc = LCCC(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_lc = LCCC(y_test, y_test_pred)

print("\nMetrics for each base model (Training and Validation):")
for model_name, metrics in all_metrics.items():
    print(f"\n{model_name} - Training Metrics:")
    print(f"  R2: {np.mean(metrics['train']['R2']):.2f} ± {np.std(metrics['train']['R2']):.2f}")
    print(f"  RMSE: {np.mean(metrics['train']['RMSE']):.2f} ± {np.std(metrics['train']['RMSE']):.2f}")
    print(f"  LCCC: {np.mean(metrics['train']['LCCC']):.2f} ± {np.std(metrics['train']['LCCC']):.2f}")
    print(f"\n{model_name} - Validation Metrics:")
    print(f"  R2: {np.mean(metrics['val']['R2']):.2f} ± {np.std(metrics['val']['R2']):.2f}")
    print(f"  RMSE: {np.mean(metrics['val']['RMSE']):.2f} ± {np.std(metrics['val']['RMSE']):.2f}")
    print(f"  LCCC: {np.mean(metrics['val']['LCCC']):.2f} ± {np.std(metrics['val']['LCCC']):.2f}")

print("\nMetrics for the ensemble model:")
print(f"Train R2: {train_r2:.2f}, Train RMSE: {train_rmse:.2f}, Train LCCC: {train_lc:.2f}")
print(f"Test R2: {test_r2:.2f}, Test RMSE: {test_rmse:.2f}, Test LCCC: {test_lc:.2f}")

# 定义保存路径
output_dir = "D:\pycharm\model\Ensemble Model"
os.makedirs(output_dir, exist_ok=True)