# IMPORTS:
import pandas as pd
import matplotlib as plt
import numpy as np
import sklearn
import math
import pickle



###############################################################################################
#                                       AUX FUNCTIONS
###############################################################################################


# ## Series complete
def series_completion(series_class, n, p, curr_set):
    '''
    complete time series to contain 30 samples
    :param series_class: current time series samples
    :param n: current number of samples in time series
    :param p: 2nd order poly fit nd array (size 3x1)
    :return: completed time series w/ 30 samples
    '''
    # calculate the number of samples to complete
    num_2_complete = 30 - n

    # generate time series for the already assigned vals
    # t_else = [x * 0.5 for x in range(num_2_complete, 30)]

    # complete the time series
    # HC - for max at the right --> complete at the left side, max at left --> complete right, otherwise both sides
    if series_class == 'r':
        # generate time series as a list:
        t = [x * 0.5 for x in range(num_2_complete+1)]
        # generate time series for the already assigned vals
        # t_else = [x * 0.5 for x in range(num_2_complete, 30)]
        # crate new series
        new_arr = np.polyval(p, t)
        old_arr = curr_set
        new_time_series = np.concatenate((new_arr, old_arr), axis=0)

    elif series_class == 'l':
        # generate time series as a list:
        t = [x * 0.5 for x in range(n)]
        # generate time series for the already assigned vals
        t_else = [x * 0.5 for x in range(n, (num_2_complete + n))]
        # crate new series
        old_arr = curr_set
        new_arr = np.polyval(p, t_else)
        new_time_series = np.concatenate((old_arr, new_arr), axis=0)

    elif series_class == 'm':
        # generate time series for the already assigned vals
        n_begin = num_2_complete / 2
        # generate time series as a list:
        t_else_begin = [x * 0.5 for x in range(n_begin)]
        t = [x * 0.5 for x in range(n_begin, (n_begin + num_2_complete))]
        t_else_end = [x * 0.5 for x in range((n_begin + num_2_complete - 1), 31)]
        # crate new series
        new_arr_1 = np.polyval(p, t_else_begin)
        old_arr = curr_set
        new_arr_2 = np.polyval(p, t_else_end)
        new_time_series = np.concatenate((new_arr_1, old_arr, new_arr_2), axis=0)

    else:
        error_handler('Falls at series completion - Bad parameters')

    return new_time_series


def error_handler(content):
    print('!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!! \n\n' + content)


###############################################################################################
#                                 model's trajectory completion
###############################################################################################
def model_traj_comp(df):

    # init new time series tables
    new_time_series_x = np.empty([1, 0], dtype=np.float32)
    new_time_series_z = np.empty([1, 0], dtype=np.float32)
    new_combined = np.empty([1, 0], dtype=np.float32)

    vel_x_aux = df.loc[df.index.str.startswith('velX')]
    vel_z_aux = df.loc[df.index.str.startswith('velZ')]

    # run over all records in the data set:
    for index in range(1, len(vel_x_aux.transpose())):

        # find the number of time stamp within current time series
        # num_time_stamps_aux = pd.notnull(vel_x_aux[index])
        try:
            num_time_stamps1 = min(np.argwhere(np.isnan(np.asarray(vel_x_aux[index], dtype=np.float32))))
        except ValueError:  # for dealing w/ the case of full data in sample
            num_time_stamps1 = [30]
        num_time_stamps = num_time_stamps1[0]
        # num_time_stamps = num_time_stamps_aux.idxmax()
        # get v_x :
        vel_x = np.asarray(vel_x_aux[index], dtype=np.float32)
        vel_x = vel_x[~np.isnan(vel_x)]
        # get v_z :
        vel_z = np.asarray(vel_z_aux[index], dtype=np.float32)
        vel_z = vel_z[~np.isnan(vel_z)]

        # if the record data is not full:
        if num_time_stamps < 30:  # TODO - check if it's 30 or 29

            # find max of v_z and its index
            # max_vel_z_val = vel_z.max(axis=1)  # search max by column
            # max_vel_z_index = vel_z.idmax(axis=1)  # search max index within row by column
            max_vel_z_index = np.argwhere(max(vel_z))
            curr_num_of_vel = num_time_stamps #len(vel_z)  # get the number of z velocities in the current row of data

            # classify to one of the three: 1.accent 2.middle phase 3.descent (depends on the maximum location)
            series_class = None
            if max_vel_z_index >= (curr_num_of_vel * 0.8):  # right
                series_class = 'r'
            elif max_vel_z_index <= (curr_num_of_vel * 0.2):  # left
                series_class = 'l'
            else:  # for the case of maximum velocity at the middle of the time series
                series_class = 'm'

            # generate time series as a list:
            t = np.arange(0, num_time_stamps/2, 0.5)
            # polyfit for x and z (2nd order) - location and velocity:
            # adjust size
            p_z = np.polyfit(t, vel_z, deg=2)
            p_x = np.polyfit(t, vel_x, deg=2)

            # # evaluate for full time series
            # vel_x = np.polyval(p, t_else_end)
            # vel_z = np.polyval(p, t_else_end)

            # complete time series:
            new_time_series_x_aux = series_completion(series_class, curr_num_of_vel, p_x, curr_set=vel_x)
            new_time_series_z_aux = series_completion(series_class, curr_num_of_vel, p_z, curr_set=vel_z)

            # append to new time series table

            # new_time_series_x = np.append(new_time_series_x, new_time_series_x_aux)
            # new_time_series_z = np.append(new_time_series_z, new_time_series_z_aux)
            new_combined = np.append(new_combined, new_time_series_x_aux)
            new_combined = np.append(new_combined, new_time_series_z_aux)
        else:  # when there's no need to add data just insert the current data to the new tables
            # new_time_series_x = np.append(new_time_series_x, vel_x)
            # new_time_series_z = np.append(new_time_series_z, vel_z)
            new_combined = np.append(new_combined, vel_x)
            new_combined = np.append(new_combined, vel_z)

    new_combined = new_combined.reshape((len(vel_x_aux.transpose())-1, 60))
    return new_time_series_x, new_time_series_z, new_combined





###############################################################################################
#                                         EM algorithm
###############################################################################################
def em_algo(df, n_components):

    # TODO - figure out which parameters to take

    df.as_matrix()

    model = sklearn.mixture.gaussian_mixture
    model.fit()





def main(full_path):

    # read csv:
    not_clean = pd.read_csv(full_path)
    transposed = not_clean.transpose()
    df = transposed.iloc[1:]  # now , df is clean

    # 1st approach:     model's trajectory completion
    vel_table_x, vel_table_z, vel_combined = model_traj_comp(df)

    # create headlines
    first_part = [('x_' + str(i)) for i in range(30)]
    sec_part = [('z_' + str(i)) for i in range(30)]
    headlines = first_part + sec_part
    # vel_combined = np.vstack((headlines, vel_combined))

    #indices = [i for i in range(28745)]
    # vel_combined = np.hstack((indices, vel_combined))

    # add headlines and conveet to data frame


    #  put vel_combined in pickle
    pck_file = open('vel_combined_test', 'wb')
    pickle.dump(pd.DataFrame(data=vel_combined, columns=headlines), pck_file)




    # # 2nd approach:     EM
    # n = np.linspace(25,50,5)
    # for n_components in n:
    #     em_algo(df, n_components)

    print('fi')
    # validation w/ KNN algorithm :
    # from sklearn.neighbors import NearestNeighbors
    # from matplotlib.colors import ListedColormap
    # from sklearn import neighbors, datasets
    # nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    # distances, indices = nbrs.kneighbors(X)

    # from sklearn.cluster import KMeans
    # flat_x = vel_table_x.flatten()
    # flat_z = vel_table_z.flatten()
    # kmeans = KMeans(n_clusters=25).fit([vel_table_x, vel_table_z])
    # kmeans = KMeans(n_clusters=25).fit(vel_table_z)
    # print('labels: \n', kmeans.labels_)
    # centers = kmeans.cluster_centers_
    # plt.figure()
    # plt.title('centroids position')
    # plt.plot(centers[0], centers[1])

if __name__ == '__main__':
    full_path = input('Enter csv file full path: ')
    main(full_path)