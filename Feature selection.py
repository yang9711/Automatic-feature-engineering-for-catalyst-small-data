import pandas as pd
import numpy as np
import random
import copy
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from warnings import simplefilter
np.set_printoptions(linewidth= 600)
simplefilter(action='ignore')

#Files
dir = "XXX/"
df = pd.read_csv(dir + "XXX.csv",index_col=0)
target="C2y"
filename= dir + "All_results.csv"
filescreen= dir + "Feature_importance.csv"

y = df[target]
df=df.drop(columns=[target])
feature_cols=df.columns.values
countlist=np.zeros_like(feature_cols)
df_count=pd.DataFrame(data=countlist,index=feature_cols, columns=["count"])
df_count=df_count.T
df_count=df_count+1.000
df_count.to_csv(filescreen)

LR_out=[]
cv=LeaveOneOut()

#GA parameters
Npop=800
Elength=8
Ngen=1600

dice=list(range(Npop))
Nelt0=Npop*2//10
Ncrs0=Npop*6//10
Nmut0=Npop*2//10
Nstr0=Npop-Nelt0-Ncrs0-Nmut0
epsilon=1.00

MAEcut=-4
scl=1.5
prc=0.02
Elmax=6
Pmax=2

scount=0
smax=10000

Com0_arr=[]
MAE0_arr=[]
LR_out=[]
MAE0_arr_max=-4

#Main body
i=0
while i < Npop:
    Com=np.random.choice(feature_cols, Elength, replace=False)
    X=df[Com]
    LR = HuberRegressor(max_iter=500, epsilon=epsilon)
    scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    MAE=np.mean(scores)
    if MAE < 0:
        Com0_arr.append(Com)
        MAE0_arr.append(MAE)
        i=i+1

print("Gen,","MaxMAE,","AvgMAE", sep="")
print("0,",np.amax(MAE0_arr),",",np.average(MAE0_arr), sep="")
df_out=[]
df_out0=pd.DataFrame(Com0_arr)
df_out1=pd.DataFrame(MAE0_arr,columns=["MAE"])
df_out=pd.concat([df_out0,df_out1],axis=1)
df_out.insert(0, "Gen", 0)
df_out.to_csv(filename, index=False)


Nelt=Nelt0
Ncrs=Ncrs0
Nmut=Nmut0
Nstr=Nstr0

for i in range(1,Ngen):
    fit=np.exp(3*(MAE0_arr-np.amin(MAE0_arr))/(np.amax(MAE0_arr)-np.amin(MAE0_arr)))
    fit=fit/sum(fit)
    MAE1_arr=[]
    Com1_arr=[]
    prio=[]
    A=np.array(MAE0_arr)
    j=0
    k1=0
    while j < Nelt:
        where=np.where(A==np.sort(A)[-(k1+1)])
        where2=where[0]
        p1=where2[0]
        Flag=0
        for k2 in range(len(Com1_arr)):
            if len(np.intersect1d(Com0_arr[p1], Com1_arr[k2])) > Elmax:
                Flag=1
        if Flag == 1:
            k1=k1+1
            continue
        Com1_arr.append(Com0_arr[p1])
        MAE1_arr.append(MAE0_arr[p1])
        j=j+1
        k1=k1+1
    if np.max(MAE1_arr) > MAE0_arr_max:
        Ncrs=Ncrs0
        Nmut=Nmut0
        Nstr=Nstr0
        scount=0
    else:
        scount=scount+1
        if scount>smax:
            Ncrs=Npop*2//10
            Nmut=Npop*5//10
            Nstr=Npop*1//10
    MAEcut=scl*np.min(MAE1_arr)
    while j < (Nelt+Ncrs):
        p1,p2=np.random.choice(dice, 2, replace=False, p=fit)
        Common=np.intersect1d(Com0_arr[p1], Com0_arr[p2])
        if len(Common) > Elength-2:
            continue
        Diff=np.setxor1d(Com0_arr[p1], Com0_arr[p2])
        Com_sub=np.random.choice(Diff, Elength-len(Common), replace=False)
        Com=np.append(Common,Com_sub)
        X=df[Com]
        LR = HuberRegressor(max_iter=500, epsilon=epsilon)
        scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        MAE=np.mean(scores)
        if MAE > MAEcut:
            df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)
            j=j+1
        else:
            df_count[Com]=df_count[Com]-df_count[Com]*prc
    while j < Nelt+Ncrs+Nmut:
        p1,p2=np.random.choice(dice, 2, replace=False, p=fit)
        Common=np.random.choice(Com0_arr[p1], Elength-1, replace=False)
        prio = df_count.loc["count"].values.tolist()
        prio=prio/np.sum(prio)
        Com_sub=np.random.choice(feature_cols, 1, p=prio)
        if len(np.intersect1d(Com0_arr[p1],Com_sub)) == 0:
            Com=np.append(Common,Com_sub)
            X=df[Com]
            LR = HuberRegressor(max_iter=500, epsilon=epsilon)
            scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            MAE=np.mean(scores)
            if MAE > MAEcut:
                df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
                Com1_arr.append(Com)
                MAE1_arr.append(MAE)
                j=j+1

            else:
                df_count[Com]=df_count[Com]-df_count[Com]*prc
    while j < Npop:
        prio = df_count.loc["count"].values.tolist()
        prio=prio/np.sum(prio)
        Com=np.random.choice(feature_cols, Elength, replace=False, p=prio)
        X=df[Com]
        LR = HuberRegressor(max_iter=500, epsilon=epsilon)
        scores = cross_val_score(LR, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        MAE=np.mean(scores)
        if MAE > MAEcut:
            df_count[Com]=df_count[Com]+(Pmax-df_count[Com])*prc
            Com1_arr.append(Com)
            MAE1_arr.append(MAE)
            j=j+1
        else:
            df_count[Com]=df_count[Com]-df_count[Com]*prc
    MAE0_arr_max=np.max(MAE0_arr)
    Com0_arr=copy.deepcopy(Com1_arr)
    MAE0_arr=copy.deepcopy(MAE1_arr)
    print(i,",",np.amax(MAE0_arr),",",np.average(MAE0_arr), sep="")
    df_out=[]
    df_out0=pd.DataFrame(Com0_arr)
    df_out1=pd.DataFrame(MAE0_arr,columns=["MAE"])
    df_out=pd.concat([df_out0,df_out1],axis=1)
    df_out.insert(0, "Gen", i)
    df_out.to_csv(filename, mode="a", header=False, index=False)
    df_count.to_csv(filescreen)


