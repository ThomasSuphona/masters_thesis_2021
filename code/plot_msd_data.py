import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import glob
import re
from natsort import natsorted


# Regualr MSD plot
def reg_msd_plot(filePath):
    list_of_files = glob.glob(filePath)
    list_of_files = natsorted(list_of_files)

    fr2sec = 0.0333
    px2m_a = 7.0137e-04

    plt.figure()
    bugTitle = ''
    for ifile, file in enumerate(list_of_files):
        msdData = scipy.io.loadmat(file)
        nameExperiment = msdData['Data'][0][0][2][0]



        # get number of bugs
        nbrActive = int(re.search('C(.*?)B', nameExperiment).group(1))

        # get number of weights
        nbrWeight = int(re.search('(.*?)W', nameExperiment).group(1))

        # get number of obstacles
        nbrObstacles = int(re.search('W(.*?)C', nameExperiment).group(1))

        bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg'%(nbrObstacles, nbrWeight*5+2)
        bugLabel = r'$N_{active}=%d$'%(nbrActive)

        weighTitle = r'$N_{passive}=%d$ and $N_{active}=%d$'%(nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg'%(nbrWeight*5+2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$'%(nbrWeight*5+2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$'%(nbrObstacles)

        msdActive = msdData['Data'][0][0][0][:, 0]/nbrActive
        msdPassive = msdData['Data'][0][0][1][:, 0]
        nbrFramesActive = msdActive.shape[0]
        nbrFramesPassive = msdPassive.shape[0]

        tauActive = np.arange(0, nbrFramesActive)
        tauPassive = np.arange(0, nbrFramesPassive)

        l = -4600

        plt.plot(tauActive[:], msdActive[:], label=bugLabel)

    plt.title(bugTitle)
    # plt.xlabel(r'$\tau$ [s]')
    # plt.ylabel(r'msd($\tau$) [m$^2$]')
    plt.xlim(0, 400)
    plt.ylim(0, 0.6)
    plt.xlabel(r'$\tau$ [timesteps]')
    plt.ylabel(r'msd($\tau$)')
    plt.legend(loc='lower right')
    plt.show()



filePath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/msd_data/0W100C*'
reg_msd_plot(filePath)
