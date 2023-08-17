import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
import sklearn.metrics as skm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

datafile="./OCM_dataset_with_1st_order_features.csv"
target="C2y"

cv=LeaveOneOut()
df = pd.read_csv(datafile, index_col=0)
y=df[target]
feature_cols=[]
feature_cols.append(["(gs_mag_moment_max)^3","1/ln(atomic_radius_rahm_min)","1/ln(en_allen_max)","1/ln(vdw_radius_ave)","ln(num_s_valence_min)","sqrt(atomic_number_std)","1/ln(dipole_polarizability_min)","sqrt(en_pauling_ave)"]) #loop0
feature_cols.append(["1/(sound_velocity_max)","1/ln(electron_affinity_ave)","1/ln(fusion_enthalpy_min)","ln(covalent_radius_pyykko_min)","(en_pauling_ave)^2","sqrt(num_s_valence_min)","1/exp(lattice_constant_std)","exp(gs_mag_moment_std)"]) #loop1
feature_cols.append(["1/(gs_est_bcc_latcnt_min)","1/ln(electron_affinity_ave)","1/sqrt(sound_velocity_max)","ln(num_s_valence_min)","sqrt(gs_mag_moment_std)","(covalent_radius_slater_ave)^3","gs_energy_ave","1/exp(covalent_radius_pyykko_triple_min)"]) #loop2
feature_cols.append(["1/(gs_est_fcc_latcnt_min)","1/ln(electron_affinity_ave)","1/ln(sound_velocity_max)","1/sqrt(covalent_radius_pyykko_min)","sqrt(hhi_r_max)","sqrt(num_s_valence_min)","gs_mag_moment_std","(atomic_radius_rahm_pro)^3"]) #loop3
feature_cols.append(["(first_ion_en_max)^3","1/ln(gs_mag_moment_min)","1/sqrt(Polarizability_min)","1/sqrt(dipole_polarizability_min)","lattice_constant_min","1/exp(electron_affinity_pro)","(gs_mag_moment_std)^2","hhi_r_max"]) #loop4

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
