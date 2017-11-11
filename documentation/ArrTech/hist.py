### instructions:
#1. choose a class, 1-25, or 0 for all classes - WARNING- for all classes takes time!!! 25 figures
#2. get the histograms for Z and X velocities, Total velocity and Z position for the chosen class


### IMPORTS:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from scipy import stats

def main(class_type,show_all=0):
    original = pd.read_csv("train.csv")
    transposed = original.transpose()
    cleaned  = transposed.iloc[1:] # get rid of redundant line
    velX = cleaned.loc[cleaned.index.str.startswith('velX')]
    # velY = cleaned.loc[cleaned.index.str.startswith('velY')]
    velZ = cleaned.loc[cleaned.index.str.startswith('velZ')]
    # posX = cleaned.loc[cleaned.index.str.startswith('posX')]
    # posY = cleaned.loc[cleaned.index.str.startswith('posY')]
    posZ = cleaned.loc[cleaned.index.str.startswith('posZ')]
    # time = cleaned.loc[cleaned.index.str.startswith('Time')]
    clas = cleaned.loc[cleaned.index.str.startswith('class')]
    clas = np.asarray(clas, dtype=np.float32)
    posZ1 = []
    velZ1 = []
    velX1 = []
    if class_type == 0 :
        for class_type in range(1,26):
            main(class_type,1)
        plt.show()
    else:
        for i in range(0,28746):
            if clas[0][i]==class_type:
                posZ1.append(posZ[i])
                velZ1.append(velZ[i])
                velX1.append(velX[i])



        plt.figure(class_type)
        plt.subplot(221)
        plt.title('hist of posZ')
        posZ1 = np.asarray(posZ1, dtype=np.float32)
        plt.hist(posZ1[~np.isnan(posZ1)],bins=100)

        plt.subplot(222)
        plt.title('hist of velZ')
        velZ1 = np.asarray(velZ1, dtype=np.float32)
        plt.hist(velZ1[~np.isnan(velZ1)],bins=100)

        plt.subplot(223)
        plt.title('hist of velX')
        velX1 = np.asarray(velX1, dtype=np.float32)
        plt.hist(velX1[~np.isnan(velX1)],bins=100)

        velTot1 = np.multiply(velX1,velX1) + np.multiply(velZ1,velZ1)
        velTot1 = np.sqrt(velTot1)

        plt.subplot(224)
        plt.title('hist of velTot')
        plt.hist(velTot1[~np.isnan(velTot1)], bins=100)

        # print "class # is: %d, format is: (D,P-value):"%(class_type)
        # print "velX: "
        # stats.kstest(velX1, 'norm')
        # print "velZ: "
        # stats.kstest(velZ1, 'norm')
        # print "velTot: "
        # stats.kstest(velTot1, 'norm')
        if show_all != 1:
            plt.show()


if __name__ == '__main__':
    class_type = input('Enter class #: ')
    main(class_type,show_all = 0)