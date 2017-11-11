import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

z = pd.read_pickle('test_pickled_data')
# print (len(z))
# z_file = 'C:\Users\spare1\PycharmProjects\Refael\test_dragZ_frame_clean'
# x_file = 'C:\Users\spare1\PycharmProjects\Refael\test_dragX_frame_clean'
# drag_z = pd.read_csv(z_file, sep=" ", header=None)
# drag_x = pd.read_csv(x_file, sep=" ", header=None)
# vel_combined = pd.read_pickle('vel_combined')

drag_z = pd.read_csv("test_dragZ_frame_clean", sep=" ", header=None)
drag_x = pd.read_csv("test_dragX_frame_clean", sep=" ", header=None)

acc_x0 = pd.read_csv('test_accX_frame0_clean', sep=" ", header=None)
acc_x1 = pd.read_csv('test_accX_frame1_clean', sep=" ", header=None)
acc_z0 = pd.read_csv('test_accZ_frame0_clean', sep=" ", header=None)
acc_z1 = pd.read_csv('test_accZ_frame1_clean', sep=" ", header=None)


drag_x.columns = ['dragX']
drag_z.columns = ['dragZ']

acc_x0.columns=['acc_x0']
acc_x1.columns=['acc_x1']
acc_z0.columns=['acc_z0']
acc_z1.columns=['acc_z1']


# In [10]: result = pd.concat([df1, df4], axis=1, join='inner')
#print(drag_x.head())
df = pd.concat([z, drag_x, drag_z, acc_x0, acc_x1, acc_z0, acc_z1], axis = 1, join = 'outer')
df.drop(df.columns[0],axis=1,inplace=True)
df = df[:-1]

# X = df.ix[:, ['avg_velX', 'avg_velZ', 'square_avg_velX', 'max_height', 'upwards',
#               'min_velZ', 'max_velZ', 'height_param', 'abs_avg_velX', 'dragZ', 'dragX']]
X = df.ix[:, ['avg_velX', 'avg_velZ', 'square_avg_velX', 'max_height', 'upwards',
              'min_velZ', 'max_velZ', 'height_param', 'abs_avg_velX', 'dragZ', 'dragX', 'acc_x0', 'acc_x1',
              'acc_z0', 'acc_z1']]

#CHANGE NAME OF MODEL TO LAST UPDATED MODEL. example: "model_with_all_acc"
file = open('model_for_test_1','rb')
model = pickle.load(file)

results = model.predict(X)
################################################################
# score for prediction
# probabilities = model.predict_proba(X)
#print (probabilities[:3,:])

# max_prob = np.amax(probabilities, axis=1)
# print (len(max_prob))
#
# partition of 3% of all samples = 8137
# after_partition = np.partition(max_prob,8137)
# partition of 4% of all samples = 10850
# after_partition = np.partition(max_prob,2715)
# partition of 10% of all samples = 27154
# after_partition = np.partition(max_prob,-27154) #upper 10%
# print (after_partition[10850-10:10850+10])
# max_threshold = after_partition[len(after_partition)-27154]
#
# max_prob_indicators = max_prob >= max_threshold

# print (low_prob_indicators)
# print (np.alen(low_prob_indicators))
# lowest_prob_classes = results[max_prob_indicators]
# z = z[:-1]
# max_prob_data = z[max_prob_indicators]
# print (results[max_prob_indicators])
# max_prob_data = pd.concat([max_prob_data,results[max_prob_indicators]], axis = 1, join = 'outer')
# print (max_prob_data)
# df.drop(df.columns[0],axis=1,inplace=True)
# print (max_prob_data)

# print (lowest_prob_classes)
# print (np.alen(lowest_prob_classes))

# class1_indicators = (results == 1)
# class2_indicators = (results == 2)
# class1_max = max_prob[class1_indicators]
# class2_max = max_prob[class2_indicators]
# plt.figure(1)
# plt.title("class1 max hist")
# plt.hist(class1_max,bins=250)
# plt.figure(2)
# plt.title("class2 max hist")
# plt.hist(class2_max,bins=250)
# plt.show()
################################################################
# CHANGE NAME
pickle.dump(results,open('test_results_morning_set1','wb'))
# print(results)

#
# pickle.dump(model, open('model_1', 'wb'))

# check the accuracy on the training set
# print(model.score(X_test, y_test))
