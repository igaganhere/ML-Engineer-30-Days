from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd

df = fetch_california_housing(as_frame=True)
df_t=df.frame.copy()
df1=df.frame.head(2000).copy()
dfc=df_t.head(2000).copy()

np.random.seed(42)

mask_medinc = np.random.rand(len(dfc)) < 0.1
mask_averooms = np.random.rand(len(dfc)) < 0.1

dfc.loc[mask_medinc, 'MedInc'] = np.nan
dfc.loc[mask_averooms, 'AveRooms'] = np.nan

#now, lets impute

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

print('\n******KNN IMPUTOR*******')
knn=KNNImputer(n_neighbors=5)
knn_imp=pd.DataFrame(knn.fit_transform(dfc),columns=df_t.columns)

true_medinc = df1.loc[mask_medinc, 'MedInc']
knn_medinc_preds = knn_imp.loc[mask_medinc, 'MedInc']
knn_mse = np.mean((true_medinc - knn_medinc_preds) ** 2)

print(f'MSE pf KNN IMPUTOR (Euclidean)-> {knn_mse}')

print('\n**********MICE*******')
mice = IterativeImputer(max_iter=10, random_state=42)
mice_imp=pd.DataFrame(mice.fit_transform(dfc), columns=df_t.columns)

mice_preds= mice_imp.loc[mask_medinc, 'MedInc']
mice_mse= np.mean((true_medinc-mice_preds)**2)

print(f'MSE of ITERATIVE IMPUTOR IS -> {mice_mse}')

