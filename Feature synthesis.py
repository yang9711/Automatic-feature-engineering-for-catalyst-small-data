import pandas as pd
def Proto(ftype,feature,x_org):
    import numpy as np
    if ftype==0:
        Fx=feature
        x=x_org
    elif ftype==1:
        Fx="1/("+feature+")"
        x=1/x_org
    elif ftype==2:
        Fx="sqrt("+feature+")"
        x=x_org**0.5
    elif ftype==3:
        Fx="1/sqrt("+feature+")"
        x=1/(x_org**0.5)
    elif ftype==4:
        Fx="("+feature+")^2"
        x=x_org**2
    elif ftype==5:
        Fx="1/("+feature+")^2"
        x=1/(x_org**2)
    elif ftype==6:
        Fx="("+feature+")^3"
        x=x_org**3
    elif ftype==7:
        Fx="1/("+feature+")^3"
        x=1/(x_org**3)
    elif ftype==8:
        Fx="ln("+feature+")"
        x=np.log(x_org)
    elif ftype==9:
        Fx="1/ln("+feature+")"
        x=1/(np.log(x_org))
    elif ftype==10:
        Fx="exp("+feature+")"
        x=np.exp(x_org)
    elif ftype==11:
        Fx="1/exp("+feature+")"
        x=1/(np.exp(x_org))
    return(Fx,x)

#Reading 0th features
dir = "XXX/"
basename = "XXXX"
df = pd.read_csv(dir + "Compound0th_" + basename + ".csv", index_col=0)

feature_cols0 = df.columns.tolist()
del feature_cols0[0]
del feature_cols0[-1]

#Synthesizing 1st featues
for i1 in range(len(feature_cols0)):
    x_temp1=df.iloc[:][feature_cols0[i1]]
    for j1 in range(12):
        Fx1,x1=Proto(j1,feature_cols0[i1],x_temp1)
        df[Fx1]=x1

df.to_csv(dir + "Compound1st_" + basename + ".csv")

#Synthesizing 2nd features
for i1 in range(len(feature_cols0)):
    x_temp1=df.iloc[:][feature_cols0[i1]]
    for i2 in range(i1+1,len(feature_cols0)):
        x_temp2=df.iloc[:][feature_cols0[i2]]
        for j1 in range(12):
            Fx1,x1=Proto(j1,feature_cols0[i1],x_temp1)
            for j2 in range(12):
                Fx2,x2=Proto(j2,feature_cols0[i2],x_temp2)
                Fx=Fx1+"|"+Fx2
                x=x1*x2
                df[Fx]=x
df.to_csv(dir + "Compound2nd_" + basename + ".csv", index=False)

