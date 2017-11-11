import pickle

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# test = pd.read_csv('test.csv')

z = pd.read_pickle('pickled_data')
#z.drop(z.columns[0],axis=1)
# vel_combined = pd.read_pickle('vel_combined')
# print (vel_combined)

# drag_z = pd.read_pickle('pickle_dragZ')
# drag_x = pd.read_pickle('pickle_dragX')
drag_x = pd.read_csv('dragX_frame.txt', sep=" ", header=None)
drag_z = pd.read_csv('dragZ_frame.txt', sep=" ", header=None)

acc_x0 = pd.read_csv('train_accX_frame0_clean', sep=" ", header=None)
acc_x1 = pd.read_csv('train_accX_frame1_clean', sep=" ", header=None)
acc_z0 = pd.read_csv('train_accZ_frame0_clean', sep=" ", header=None)
acc_z1 = pd.read_csv('train_accZ_frame1_clean', sep=" ", header=None)


drag_x.columns=['dragX']
drag_z.columns=['dragZ']

acc_x0.columns=['acc_x0']
acc_x1.columns=['acc_x1']
acc_z0.columns=['acc_z0']
acc_z1.columns=['acc_z1']

# In [10]: result = pd.concat([df1, df4], axis=1, join='inner')
# print(drag_x.head())
df = pd.concat([z, drag_x, drag_z, acc_x0, acc_x1, acc_z0, acc_z1], axis = 1, join = 'outer')
df = df[:-1]
df.drop(df.columns[0],axis=1,inplace=True)
# vel_combined.drop(vel_combined.columns[30:59],axis=1,inplace=True)
# print (df)

X = df.ix[:, ['avg_velX', 'avg_velZ', 'square_avg_velX', 'max_height', 'upwards',
              'min_velZ', 'max_velZ', 'height_param', 'abs_avg_velX', 'dragZ', 'dragX', 'acc_x0', 'acc_x1',
              'acc_z0', 'acc_z1']]
# X = pd.concat([X, vel_combined], axis = 1, join = 'outer')
# print (X)
# X = df.ix[:, ['avg_velX', 'avg_velZ', 'square_avg_velX', 'max_height', 'upwards',
#               'min_velZ', 'max_velZ', 'height_param', 'abs_avg_velX']]

y = df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=20)

# model = LogisticRegression(multi_class='ovr')
model = RandomForestClassifier(n_estimators=70, max_depth=25)

# model = model.fit(X_train, y_train)
# check the accuracy on the training set
# print(model.score(X_test, y_test))


# RUN THIS ONLY BEFORE RUNNING ON TEST
model = model.fit(X, y)

pickle.dump(model, open('model_for_test_1', 'wb'))

# check the accuracy on the training set
#print(model.score(X_test, y_test))
print("WARNING: whatever")
