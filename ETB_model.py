import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
import sklearn.metrics as skm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

datafile="./ETB_dataset_with_1st_order_features.csv"
target="C4H6 yield (%)"

cv=LeaveOneOut()
df = pd.read_csv(datafile)
y=df[target]
feature_cols=[]
feature_cols.append(["1/(gs_energy_var)","1/(hhi_p_avg)","1/sqrt(first_ion_en_var)","ln(sound_velocity_ssd)","thermal_conductivity_prod","1/(boiling_point_avg)^3","1/(vdw_radius_alvarez_gmean)^2","1/(period_avg)^2"])

out=[target]
LR = HuberRegressor(max_iter=20000, epsilon=1.00)

Loop=0
for i in range(len(feature_cols)):
    Com=[]
    Com=feature_cols[i]
    X=df[Com]
    LR.fit(X,y)
    y_pred = LR.predict(X)
    cross_val_scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error")
    cross_val_scores_isnan = cross_val_scores[~np.isnan(cross_val_scores)]
    print(Loop,skm.mean_absolute_error(y,y_pred),-np.average(cross_val_scores_isnan))
    Loop=Loop+1
    temp="pred-" + str(i)
    df[temp]=y_pred
    out.append(temp)
df[out].to_csv("./out.csv")

