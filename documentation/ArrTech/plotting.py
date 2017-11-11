### IMPORTS:
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main(index):
    original = pd.read_csv("train.csv")
    transposed = original.transpose()
    cleaned  = transposed.iloc[1:] # get rid of redundant line
    velX = cleaned.loc[cleaned.index.str.startswith('velX')]
    velY = cleaned.loc[cleaned.index.str.startswith('velY')]
    velZ = cleaned.loc[cleaned.index.str.startswith('velZ')]
    posX = cleaned.loc[cleaned.index.str.startswith('posX')]
    posY = cleaned.loc[cleaned.index.str.startswith('posY')]
    posZ = cleaned.loc[cleaned.index.str.startswith('posZ')]
    time = cleaned.loc[cleaned.index.str.startswith('Time')]

    print "plotting index: %d" % (index)

### Plot
    plt.figure(1)
    plt.subplot(311)
    plt.title('position- Z as a function of X')
    plt.plot(posX[index],posZ[index])
    plt.subplot(312)
    plt.title('position- Z as a function of Y')
    plt.plot(posY[index],posZ[index])
    plt.subplot(313)
    plt.title('position- Y as a function of X')
    plt.plot(posX[index],posY[index])

    plt.figure(2)
    plt.title('position- Z and y as a function of X')
    plt.plot(posX[index],posZ[index])
    plt.plot(posX[index],posY[index] + np.mean(posZ[index]))

    plt.figure(3)
    plt.subplot(311)
    plt.title('velocity- X as a function of time')
    plt.plot(time[index],velX[index])
    plt.subplot(312)
    plt.title('velocity- Y as a function of time')
    plt.plot(time[index],velY[index])
    plt.subplot(313)
    plt.title('velocity- Z as a function of time')
    plt.plot(time[index],velZ[index])

    plt.figure(4)
    plt.subplot(211)
    plt.title('position- X as a function of time')
    plt.plot(time[index], posX[index])
    plt.subplot(212)
    plt.title('position- Z as a function of time')
    plt.plot(time[index], posZ[index])


    plt.show()
if __name__ == '__main__':
    index = input('Enter  sample #: ')
    main(index)