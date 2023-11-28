# from PyEMD import EMD
# import numpy as np
# import pandas as pd
# from scipy.spatial.distance import cosine
#
# emd = EMD()
# IMFs_list = []
# df = pd.read_csv("../dataset/American/train_data.csv")
# df = df.dropna()
# # 创建EMD对象
# emd = EMD()
#
# # 选择要分析的特征列
# features_to_analyze = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
#
# # 分解每个特征列并将IMF存储在新的数据框中
# imf_df = pd.DataFrame()
# for feature in features_to_analyze:
#     imfs = emd(df[feature].values)
#     print(imfs)
#     for i, imf in enumerate(imfs):
#         imf_df[f'{feature}_IMF{i+1}'] = imf
#         # print(i)
#         # print(imf)
# correlation_matrix = imf_df.corr()
from PyEMD import EMD
import numpy as np

