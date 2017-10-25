#!/usr/bin/python

# small script to add speed, acceleration features
# TODO: vectors logic needs to be reviewed


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.metrics import confusion_matrix

import random

random.seed(12345)
pd.options.display.max_columns = 100


import itertools


def add_velocities(data):
    for i in range(30):
        vx = data['velX_%d' % i]
        vy = data['velY_%d' % i]
        vz = data['velZ_%d' % i]

        vxy = vx+vy
        data['velXY_%d' % (i)] = vxy

        vxz = vx+vz
        data['velXZ_%d' % (i)] = vxz

        vyz = vy + vz
        data['velYZ_%d' % (i)] = vyz

        vxyz = vx + vy + vz
        data['velXYZ_%d' % (i)] = vxyz

    return data


def add_acceleraion(data):
    acel_dims = ['X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ']
    for dim in acel_dims:
        for i in range(0, 30-1):
            v_dim = data['vel%s_%d' % (dim, i)]   # e.g. vzlX_10, velYZ_20
            v_dim_plus_1 = data['vel%s_%d' % (dim, i+1)]

            accel_dim = v_dim_plus_1 - v_dim
            data['accel_%s_%d' % (dim,i)] = accel_dim

    return data


def main():
    path = './'
    name = ''
    out_name = path + name + 'submission_py.csv'
    train_data = pd.read_csv(path + name + '../train_sample.csv')
    print "after loading data from file, features: ", train_data.shape

    train_data = add_velocities(train_data)
    print "after adding velocities features: ", train_data.shape

    train_data = add_acceleraion(train_data)
    print "after adding acceleration features: ", train_data.shape

    for col in train_data.columns:
        print col


if __name__ == '__main__':
    main()