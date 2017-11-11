import pandas as pd

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

original  = pd.read_csv('test.csv')

transposed = original.transpose()
velocities = transposed.iloc[1:]  # get rid of redundant line
velocities = velocities.loc[velocities.index.str.startswith('velX')]
velocities = velocities.transpose()
# velocities.fillna(0, inplace = True)
velocities['avg'] = velocities.mean(axis=1, skipna=True)
velocities['var'] = velocities.var(axis=1, skipna=True)

z_velocities = transposed.iloc[1:]  # get rid of redundant line
z_velocities = z_velocities.loc[z_velocities.index.str.startswith('velZ')]
z_velocities = z_velocities.transpose()
z_velocities['avg'] = z_velocities.mean(axis=1, skipna=True)
z_velocities['var'] = z_velocities.var(axis=1, skipna=True)
z_velocities['max'] = z_velocities.max(axis=1, skipna=True)
z_velocities['min'] = z_velocities.min(axis=1, skipna=True)
# print(z_velocities.heads(300))


heights = transposed.iloc[1:]  # get rid of redundant line
heights = heights.loc[heights.index.str.startswith('posZ')]
heights= heights.transpose()
heights['max'] = heights.max(axis=1, skipna=True)

df = pd.DataFrame(original)
df['avg_velX'] = velocities['avg']
df['square_avg_velX']=df['avg_velX']**2
df['avg_velZ'] = z_velocities['avg']
df['max_velZ'] = z_velocities['max']
df['min_velZ'] = z_velocities['min']
df['upwards'] = df['avg_velZ'] > 0
df['max_height']= heights['max']
df['abs_avg_velX'] = abs(df.avg_velX)
# z = df.groupby('class').apply(lambda x: x['max_height'].max()).to_frame()
# z.columns = ['max_class_height']
# print(z.head())
# df = pd.merge(df,z, how= 'left',left_on="class" , right_index=True)

df['height_p']=df.max_velZ.convert_objects(convert_numeric=True) * df.max_height.convert_objects(convert_numeric=True)
df['height_param']=df.height_p / 1000
df.to_pickle('test_pickled_data')

# X= df.ix[:, ['avg_velX', 'avg_velZ', 'square_avg_velX', 'max_height', 'upwards',
#              'min_velZ', 'max_velZ', 'height_param', 'abs_avg_velX']]
# # print(X[:100])
# y = original['class']
# # print(y[:100])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state=25)
#
#
# # model = LogisticRegression(multi_class='ovr')
# model = KMeans (n_clusters=25)
# model = model.fit(X_train, y_train)
#
# # check the accuracy on the training set
# print(model.score(X_test, y_test))
#
