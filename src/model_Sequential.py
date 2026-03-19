# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import median_absolute_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
# %%
df= pd.read_csv(r'..\data\processed\df_preprocessed.csv')
df = df.drop(columns=['profit_per_unit', 'sales_per_unit' ])
df.info()
@tf.keras.utils.register_keras_serializable()
def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - ss_res / (ss_tot + tf.keras.backend.epsilon()))
# %%
#Sequential
#separere 10% din anul 2017 pt pred blind
df_2017 = df[df['order_year'] == 2017]
df_before_2017 = df[df['order_year'] < 2017]
df_blind_pred = df_2017.sample(frac=0.1, random_state=42)
df_work = pd.concat([df_before_2017, df_2017.drop(df_blind_pred.index)])

categorical_cols = ['ship_mode', 'segment', 'region', 'category', 'sub-category', 'product_id', 'product_name', 'state', 'city', 'postal_code', 'ship_mode']
numerical_cols = ['sales', 'quantity', 'discount', 'order_month', 'order_day', 'ship_day']
target = 'profit'
df_processed = pd.get_dummies(df[categorical_cols + numerical_cols + [target]], columns=categorical_cols)


X_work = df_processed.drop(columns=[target]).loc[df_work.index]
y_work = df_processed[target].loc[df_work.index]

X_blind = df_processed.drop(columns=[target]).loc[df_blind_pred.index]
y_blind_actual = df_processed[target].loc[df_blind_pred.index]

X_train, X_test, y_train, y_test = train_test_split(X_work, y_work, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_blind_scaled = scaler.transform(X_blind)


model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(250, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', r_squared])

early_stop = EarlyStopping(
    monitor='val_mae',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print(f"Antrenare pe {X_train.shape[0]} randuri cu Early Stopping (patience=5)")

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)



y_blind_pred = model.predict(X_blind_scaled)
print(f"Median MAE pe Blind Set: {median_absolute_error(y_blind_actual, y_blind_pred):.2f}")
print(f"R2 pe Blind Set: {r2_score(y_blind_actual, y_blind_pred):.2f}")
# %%
# evaluare model
eval_results = model.evaluate(X_test_scaled, y_test, verbose=0)
loss_test, mae_test, r2_test_metric = eval_results

print(f"\n{' METRICI SET TEST (Validation) ':=^40}")
print(f"MAE: {mae_test:.4f}")
print(f"R2 Score (din metrica): {r2_test_metric:.4f}")


predictions_blind = model.predict(X_blind_scaled)


mae_blind = mean_absolute_error(y_blind_actual, predictions_blind)
m_mae_blind = median_absolute_error(y_blind_actual, predictions_blind)
r2_blind = r2_score(y_blind_actual, predictions_blind)

print(f"\n{' METRICI SET BLIND (Predictie 2017) ':=^40}")
print(f"MAE pe set blind: {mae_blind:.4f}")
print(f"R2 Score pe set blind: {r2_blind:.4f}")


results_comparison = pd.DataFrame({
    'Real': y_blind_actual.values,
    'Predicție': predictions_blind.flatten(),
    'Eroare Absolută': abs(y_blind_actual.values - predictions_blind.flatten())
})

print("\nPrimele 10 predictii pe setul Blind:")
print(results_comparison.head(10))
# %%
#MLPRegressor
le = LabelEncoder()
categorical_features = ['ship_mode', 'segment', 'region', 'category', 'sub-category', 'product_id', 'product_name', 'state', 'city', 'postal_code', 'ship_mode']
for col in categorical_features:
    df[col] = le.fit_transform(df[col].astype(str))

features_ml = ['ship_mode', 'segment', 'region', 'category', 'sub-category', 'product_id', 'product_name', 'state', 'city', 'postal_code', 'ship_mode']
X = df[features_ml]
y = df['profit']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-4,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
m_mae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Iteratii pana la oprire: {model.n_iter_}")
print(f"Best MAE: {mae:.2f}")
print(f"Best Median MAE: {m_mae:.2f}")
print(f"Best R2 Score: {r2:.4f}")
# %%
# ANALIZA SI EVIDENTIERE OUTLIER (BE-11335)
client_stats = df.groupby('customer_id')['profit'].sum()
mean_p = client_stats.mean()
std_p = client_stats.std()
be_profit = client_stats.get('BE-11335', 0)
z_score_be = (be_profit - mean_p) / std_p


df_filtered = df[df['customer_id'] != 'BE-11335']

df_2017 = df_filtered[df_filtered['order_year'] == 2017]
df_before_2017 = df_filtered[df_filtered['order_year'] < 2017]
df_blind_raw = df_2017.sample(frac=0.1, random_state=42)
df_work_raw = pd.concat([df_before_2017, df_2017.drop(df_blind_raw.index)])


categorical_cols = ['ship_mode', 'segment', 'region', 'category', 'sub-category', 'product_id', 'product_name', 'state', 'city', 'postal_code', 'ship_mode']
    #['ship_mode', 'segment', 'region', 'category', 'sub-category', 'state']
numerical_cols = ['sales', 'quantity', 'discount', 'order_month', 'order_day', 'ship_day']
target = 'profit'

df_processed = pd.get_dummies(df_filtered[categorical_cols + numerical_cols + [target]], columns=categorical_cols)


X_work = df_processed.drop(columns=[target]).loc[df_work_raw.index]
y_work = df_processed[target].loc[df_work_raw.index]
X_blind = df_processed.drop(columns=[target]).loc[df_blind_raw.index]
y_blind = df_processed[target].loc[df_blind_raw.index]


X_train, X_test, y_train, y_test = train_test_split(X_work, y_work, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_blind_scaled = scaler.transform(X_blind)


model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(1000, activation='relu'),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', r_squared])
early_stop = EarlyStopping(monitor='val_mae', patience=5, restore_best_weights=True, verbose=1)

model.fit(X_train_scaled, y_train,
          validation_data=(X_test_scaled, y_test),
          epochs=100,
          batch_size=32,
          callbacks=[early_stop],
          verbose=1)


pred_test = model.predict(X_test_scaled).flatten()
pred_blind = model.predict(X_blind_scaled).flatten()


def get_metrics(actual, pred):
    return [mean_squared_error(actual, pred), mean_absolute_error(actual, pred),
            median_absolute_error(actual, pred), r2_score(actual, pred)]

m_test = get_metrics(y_test, pred_test)
m_blind = get_metrics(y_blind, pred_blind)


print(f"\n{' DIAGNOSTIC OUTLIER: BE-11335 ':=^60}")
print(f"Profit Total Client:     {be_profit:>10.2f}")
print(f"Media Profit/Client:     {mean_p:>10.2f}")
print(f"Z-Score (Deviatii Std):  {z_score_be:>10.2f} (Scor < -3 = Outlier Extrem)")
print(f"Status:                  EXCLUS din antrenare pentru stabilitate.")

print(f"\n{' RAPORT COMPARATIV PERFORMANTA ':=^60}")
report_df = pd.DataFrame({
    'Metrica': ['MSE', 'MAE', 'Median MAE', 'R2 Score'],
    'Set TEST (Validare)': m_test,
    'Set BLIND (2017)': m_blind
})
print(report_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\n{' ANALIZA DE STABILITATE ':=^60}")
diff_r2 = m_test[3] - m_blind[3]
status_r2 = "model STABIL (NU avem Overfitting!) " if diff_r2 < 0.1 else "OVERFITTING DETECTAT"
print(f"Variatie R2 Score: {diff_r2:.4f} -> Model {status_r2}")
print("="*60)

df_blind_raw['profit_predicted'] = pred_blind

df_blind_raw['error_abs'] = abs(df_blind_raw['profit_predicted'] - df_blind_raw['profit'])


print("\nPrimele 5 rânduri din Blind Set cu predicții:")
print(df_blind_raw[['customer_id', 'profit', 'profit_predicted', 'error_abs']].head())

df_blind_raw.to_csv('../data/data_output/predictii_profit_sequential_blind_2017.csv', index=False)

model.save('../models/profit_prediction_Sequential_draft.keras')


# %%
# evaluare model
eval_results = model.evaluate(X_test_scaled, y_test, verbose=0)
loss_test, mae_test, r2_test_metric = eval_results
profit_mediu = df['profit'].mean()
profit_median = df['profit'].median()
m_med_test = median_absolute_error(y_test, pred_test)

print(f"\n{' METRICI SET TEST (Validation) ':=^40}")
print(f"MAE: {mae_test:.4f}")
print(f"MAE median: {m_med_test:.4f}")
print(f"Profit_mediu: {profit_mediu:.4f}")
print(f"Profit_median: {profit_median:.4f}")
print(f"R2 Score (din metrica): {r2_test_metric:.4f}")


predictions_blind = model.predict(X_blind_scaled)


mae_blind = mean_absolute_error(y_blind, predictions_blind)
m_mae_blind = median_absolute_error(y_blind, predictions_blind)
r2_blind = r2_score(y_blind, predictions_blind)
profit_mediu_blind = df_blind_raw['profit'].mean()
profit_median_blind = df_blind_raw['profit'].median()

print(f"\n{' METRICI SET BLIND (Predictie 2017) ':=^40}")
print(f"MAE pe set blind: {mae_blind:.4f}")
print(f"MAE MED pe set blind: {m_mae_blind:.4f}")
print(f"Profit_mediu: {profit_mediu_blind:.4f}")
print(f"Profit_median: {profit_median_blind:.4f}")
print(f"R2 Score pe set blind: {r2_blind:.4f}")


results_comparison = pd.DataFrame({
    'Profit Real': y_blind.values,
    'Profit Prezis': predictions_blind.flatten(),
    'Eroare Absoluta': abs(y_blind.values - predictions_blind.flatten())
})

print("\nPrimele 50 predictii pe setul Blind:")
print(results_comparison.head(50))