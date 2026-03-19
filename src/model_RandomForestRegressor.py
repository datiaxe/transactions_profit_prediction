# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

raw_data = pd.read_csv('df_preprocessed.csv')
# %%
cust_features = raw_data.groupby('customer_id').agg({
    'sales': ['sum', 'mean', 'std'],
    'discount': ['mean', 'max'],
    'quantity': 'sum',
    'shipping_delay': 'mean',
    'order_id': 'nunique',
    'profit': 'sum'
}).reset_index()


cust_features.columns = [
    'customer_id', 'total_sales', 'avg_sales', 'std_sales',
    'avg_discount', 'max_discount', 'total_quantity',
    'avg_shipping_delay', 'order_frequency', 'total_profit'
]

cust_features = cust_features.fillna(0)


segments = raw_data.groupby('customer_id')['segment'].first().reset_index()
cust_features = cust_features.merge(segments, on='customer_id')


encoder = OneHotEncoder(sparse_output=False)
segment_encoded = encoder.fit_transform(cust_features[['segment']])
segment_df = pd.DataFrame(segment_encoded, columns=encoder.get_feature_names_out(['segment']))
cust_features = pd.concat([cust_features, segment_df], axis=1).drop(columns=['segment'])


X = cust_features.drop(columns=['customer_id', 'total_profit'])
y = cust_features['total_profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.4f}")