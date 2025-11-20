import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from data_constants import OBS_DATA_DIR

CLEAN_DATA_PATH  = f"{OBS_DATA_DIR}/clean/20220101_df_clean.csv"
np.random.seed(2)

if __name__ == '__main__':

    # Import cleaned dataset
    df = pd.read_csv(CLEAN_DATA_PATH)
    # print(df.isna().any())
    print(df['target_y'])
    print(df['TEMP'])

    # PREPROCESSING
    X = df[['TEMP', 'RELATIVE_HUMIDITY', 'PRECIP_AMOUNT', 'VISIBILITY', 'cos_day', 'cos_time']]
    y = df['target_y'].values.reshape(-1,1)
    print('original: ', X)

    # Create a heatmap using seaborn
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()

    X_scaled = StandardScaler().fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    print(f"X_scaled: {X_scaled.shape}, {X_scaled}")
    print(f"y: {y.shape}, {y}")
    print('\n')

    # STATSMODELS
    model = sm.OLS(endog=y, exog=X_scaled)
    res = model.fit()
    print(res.summary())


    # PLOT PREDICTIONS
    y_hat = res.predict()
    print(f"y_hat: {y_hat.shape}, {y_hat}")

    plt.scatter(df.julian_day, y_hat, color='r', label='predictions')
    plt.scatter(df.julian_day, y, label='observations')
    plt.legend(loc='upper right')
    plt.xlabel('julian day')
    plt.ylabel('beta')

    plt.show()

    # SCIKIT LEARN

    # lr = LinearRegression()
    # lr.fit(X=X_scaled, y=y)
    # print(' - coeffs: ', lr.coef_)
    # print(' - intercept: ', lr.intercept_)

    # Normalize data


