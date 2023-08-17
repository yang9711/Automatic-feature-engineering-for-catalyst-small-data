import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
import sklearn.metrics as skm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

datafile="./TWC_dataset_with_1st_order_features.csv"
target="T50(NO)"

cv=LeaveOneOut()
df = pd.read_csv(datafile)
y=df[target]
feature_cols=[]
feature_cols.append(["(en_pauling_max)^3","1/ln(icsd_volume_min)","1/ln(sound_velocity_var)","sqrt(hhi_p_var)","ln(thermal_conductivity_gmean)","dipole_polarizability_var","vdw_radius_alvarez_avg","exp(vdw_radius_alvarez_avg)"])

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

