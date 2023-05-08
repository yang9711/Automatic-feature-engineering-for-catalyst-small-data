import numpy as np
import pandas as pd

dir = "XXX/"
basename = "XXX"

Flag = 1
# 0:Total loading is variable.
# 1:Total loading is constant.

# Data loading
df_descriptor = pd.read_csv(dir + "xenonpy_normalized.csv", index_col=0)
df_perform = pd.read_csv(dir + basename + ".csv")
df_perform = df_perform.drop(df_perform.columns[0], axis=1)
print(df_perform)

desc_c = df_descriptor.columns.tolist()
perf_c = df_perform.columns.tolist()
target = df_perform.iloc[:, -1]
df_perform = df_perform.drop(df_perform.columns[-1], axis=1)
perform = df_perform.values
df_de_extract = df_descriptor.loc[perf_c[0:-1]].T
de_extract = df_de_extract.values


# 0th feature assignment
result = []
column = []
for i in range(len(perform)):
    buffer1 = []
    buffer2 = []
    p_sum = sum(perform[i])
    for j in range(len(de_extract)):
        pro =  perform[i] * de_extract[j]
        p_max = np.max(np.array(de_extract[j])[np.nonzero(np.array(perform[i]))])
        p_min = np.min(np.array(de_extract[j])[np.nonzero(np.array(perform[i]))])
        p_w_sum = sum(pro)
        p_w_avg = p_w_sum / p_sum
        p_ssd = sum((de_extract[j] - np.mean(de_extract[j])) ** 2 * perform[i])
        p_w_var = p_ssd / p_sum
        p_w_prod = np.prod(de_extract[j] ** perform[i])
        p_w_gmean = p_w_prod ** (1 / p_sum)
        if Flag == 0:
            buffer1.extend([p_min, p_max, p_w_sum, p_w_avg, p_ssd, p_w_var, p_w_prod, p_w_gmean])
            buffer2.extend([desc_c[j] + "_min", desc_c[j] + "_max", desc_c[j] + "_sum", desc_c[j] + "_avg",
                            desc_c[j] + "_ssd", desc_c[j] + "_var", desc_c[j] + "_prod", desc_c[j] + "_gmean"])
        else:
            buffer1.extend([p_min, p_max, p_w_avg, p_w_var, p_w_gmean])
            buffer2.extend([desc_c[j] + "_min", desc_c[j] + "_max", desc_c[j] + "_avg",
                            desc_c[j] + "_var", desc_c[j] + "_gmean"])
    result.append(buffer1)
    column.append(buffer2)
df_result = pd.DataFrame(result, columns=column[0])


# Remove 0 variance and missing features
df_result = df_result.loc[:, (df_result != df_result.iloc[0]).any()]
df_result = df_result.dropna(axis=1)

# Treatment of 0 and 1
for i in df_result.columns:
    min_second = np.min(df_result[i].iloc[np.nonzero(df_result[i].values)]) * 10 ** -4
    df_result[i] = df_result[i] + min_second

# Outputing
df_out = pd.concat([df_result, target], axis=1)
df_out.to_csv(dir + "Compound0th_" + basename + ".csv")


