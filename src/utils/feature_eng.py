import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def calculate_spectral_indices(df):
    indices = {
        'NDVI': ('b02', 'b01'),  # Normalized Difference Vegetation Index
        'NDWI': ('b02', 'b06'),  # Normalized Difference Water Index
        'NDMI': ('b02', 'b06'),  # Normalized Difference Moisture Index
    }
    
    # Calculate normalized difference indices
    for name, (band1, band2) in indices.items():
        b1 = f'modis_sur_refl_{band1}_1'
        b2 = f'modis_sur_refl_{band2}_1'
        df[name] = (df[b1] - df[b2]) / (df[b1] + df[b2])
    
    # Calculate other specialized indices
    df['FDI'] = df['modis_sur_refl_b03_1'] - (df['modis_sur_refl_b02_1'] + 
                (df['modis_sur_refl_b04_1'] - df['modis_sur_refl_b02_1']) * (3 - 2) / (4 - 2))
    
    df['EVI'] = 2.5 * ((df['modis_sur_refl_b02_1'] - df['modis_sur_refl_b01_1']) / 
                (df['modis_sur_refl_b02_1'] + 6 * df['modis_sur_refl_b01_1'] - 
                7.5 * df['modis_sur_refl_b03_1'] + 1))
    
    df['SABI'] = (df['modis_sur_refl_b02_1'] - df['modis_sur_refl_b04_1']) / \
                 (df['modis_sur_refl_b03_1'] + df['modis_sur_refl_b01_1'])
    
    df['NIRPI'] = df['modis_sur_refl_b02_1'] / (df['modis_sur_refl_b02_1'] + df['modis_sur_refl_b01_1'])
    
    df['WRI'] = (df['modis_sur_refl_b04_1'] - df['modis_sur_refl_b01_1']) / \
                (df['modis_sur_refl_b02_1'] + df['modis_sur_refl_b06_1'])
    
    return df

def add_polynomial_features(X, importance_df, top_n=3, update_features=False):
    global feature_cols
    X = np.array(X)
    top_features = importance_df.nlargest(top_n, 'Importance')['Feature'].tolist()
    top_features_idx = [feature_cols.index(f) for f in top_features]
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[:, top_features_idx])
    X_poly = X_poly[:, top_n:]  # Remove linear terms
    
    # Generate polynomial feature names
    poly_feature_names = [
        f"{top_features[i]}*{top_features[j]}" if i != j else f"{top_features[i]}^2"
        for i in range(len(top_features))
        for j in range(i, len(top_features))
    ]
    
    if update_features:
        feature_cols.extend(poly_feature_names)
    
    return np.hstack([X, X_poly])