"""Retrain the Bangalore home price model and save artifacts."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import json
import os

# Load data
df1 = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Bengaluru_House_Data.csv'))
print(f"Loaded {df1.shape[0]} rows")

# Drop unnecessary columns
df2 = df1.drop(['area_type', 'availability', 'society', 'balcony'], axis='columns')
df3 = df2.dropna()

# Extract BHK from size
df3 = df3.copy()
df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

# Convert total_sqft to numeric
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

# Price per sqft
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

# Group rare locations into 'other'
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Remove outliers: sqft per BHK < 300
df6 = df5[~(df5.total_sqft / df5.BHK < 300)]

# Remove price_per_sqft outliers per location (mean +/- 1 std)
def remove_outlier(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_outlier(df6)

# Remove BHK outliers
def remove_bhk_outlier(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outlier(df7)

# Remove bathroom outliers
df9 = df8[df8.bath < df8.BHK + 2]

# Prepare final dataframe
df10 = df9.drop(['price_per_sqft', 'size'], axis='columns')

# One-hot encode location
dummies = pd.get_dummies(df10.location, dtype=int)
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

# Split features and target
X = df12.drop('price', axis='columns')
y = df12.price

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=10)
lr = LinearRegression()
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print(f"Model R² score: {score:.4f}")

# Save artifacts
artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'server', 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)

with open(os.path.join(artifacts_dir, 'banglore_home_prices_model.pickle'), 'wb') as f:
    pickle.dump(lr, f)

columns = {'data_columns': [col.lower() for col in X.columns]}
with open(os.path.join(artifacts_dir, 'banglore_home_prices_model_columns.json'), 'w') as f:
    json.dump(columns, f)

print(f"Saved model with {len(columns['data_columns'])} features to {artifacts_dir}")
print("Test prediction (1st Phase JP Nagar, 1000sqft, 3bhk, 3bath):")

# Quick test
from importlib import reload
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'server'))
import util
util.load_saved_artifacts()
print(f"  Price: {util.get_estimated_price('1st phase jp nagar', 1000, 3, 3)} Lakhs")
