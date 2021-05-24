import numpy as np
import glob
import scipy.io as scio
from astropy.visualization import hist
import matplotlib.pyplot as plt
from natsort import natsorted
import seaborn as sns
import pandas as pd

def read_data(listOfFiles):

    angularVelocityDict = {}
    for iFile, file in enumerate(listOfFiles):
        data = scio.loadmat(file)

        # Load raw data
        videoName = data['Data'][0, 0][0][0]
        frameRate_Hz = data['Data'][0, 0][1][0][0]
        duration_s = data['Data'][0, 0][2][0][0]
        nbrFrames = data['Data'][0, 0][3][0][0]
        frameHeight_px = data['Data'][0, 0][4][0][0]
        frameWidth_px = data['Data'][0, 0][5][0][0]
        sizeActive_m = data['Data'][0, 0][6][0][0]
        sizePassive_m = data['Data'][0, 0][7][0][0]
        pixelsToMeterPassive = data['Data'][0, 0][8][0][0]
        nbrPassiveTrue = data['Data'][0, 0][9][0][0]
        nbrPassiveTracked = data['Data'][0, 0][10][0][0]
        passiveTrajectoriesX_px = data['Data'][0, 0][11]
        passiveTrajectoriesY_px = data['Data'][0, 0][12]
        activeTrajectoriesX_px = data['Data'][0, 0][13]
        activeTrajectoriesY_px = data['Data'][0, 0][14]
        pixelsToMeterActive = data['Data'][0, 0][15][0][0]
        nbrActiveTrue = data['Data'][0, 0][16][0][0]
        nbrActiveTracked = data['Data'][0, 0][17][0][0]
        nbrWeights = data['Data'][0, 0][18][0][0]
        nbrWeights_kg = data['Data'][0, 0][19][0][0]
        passiveIndices = data['Data'][0, 0][20][:, 0]-1
        activeIndices = data['Data'][0, 0][21][:, 0]-1
        activeOrientation = data['Data'][0, 0][22]
        nameExperiment = f'{nbrWeights}W{nbrPassiveTrue}C{nbrActiveTrue}B'


        nbrPassiveUse = len(passiveIndices)
        nbrActiveUse = len(activeIndices)

        #Process and convert to SI units
        #pTrajX = passiveTrajectoriesX_px[passiveIndices, :].toarray()*pixelsToMeterPassive
        #pTrajY = passiveTrajectoriesY_px[passiveIndices, :].toarray()*pixelsToMeterPassive

        #aTrajX = activeTrajectoriesX_px[activeIndices, :].toarray()*pixelsToMeterActive
        #aTrajY = activeTrajectoriesY_px[activeIndices, :].toarray()*pixelsToMeterActive

        #Not sure what unit the orientation is in
        orientation = activeOrientation.toarray()
        nbrAngularVelocityPoints = np.count_nonzero(orientation) - orientation.shape[0]
        angularVelocity = np.zeros(nbrAngularVelocityPoints)

        lower = 0
        upper = 0

        for iActiveParticle in range(nbrActiveUse):
            orientationNonzero_i = orientation[iActiveParticle, np.nonzero(orientation[iActiveParticle, :])][0]

            dTheta_i = np.asarray([(l - m) for (l, m) in zip(orientationNonzero_i[1:], orientationNonzero_i)])

            upper = len(dTheta_i) + lower

            #Assume that the original angles was given in degrees
            angularVelocity[lower:upper] = dTheta_i*(np.pi/180)*frameRate_Hz

            lower = upper

        angularVelocityDict[nameExperiment] = (nbrWeights, nbrPassiveTrue, nbrActiveTrue, angularVelocity)

    return angularVelocityDict

def plot_angularvelocity_histogram(angularVelocity, N):


    nameExperiments = list(angularVelocity.keys())

    count = 0

    d = {}

    for iexp, experiment in enumerate(nameExperiments[:]):
    #for experiment in nameExperiments:
        angVel = angularVelocity[experiment][3]
        #angVel = angVel[np.nonzero(angVel)]
        #angVel = angVel[angVel >= 0.00001]
        w = int(angularVelocity[experiment][0])
        c = int(angularVelocity[experiment][1])
        b = int(angularVelocity[experiment][2])

        bugLabel = r'$N_{active}=%d$'%(b)
        obstacleLabel = r'$N_{passive}=%d$' % (c)
        weightLabel = r'$m_{passive}=%d\cdot10^{-3}$kg'%(5*w+2)

        bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg'%(c, 5*w+2)
        weightTitle = r'$N_{passive}=%d$, $N_{active}=%d$'%(c, b)
        obstacleTitle = r'$N_{active}=%d$, $m_{passive}=%d\cdot10^{-3}$kg' % (b, 5 * w + 2)

        if N == 0:
            label = weightLabel
            title = weightTitle
        elif N == 1:
            label = obstacleLabel
            title = obstacleTitle
        elif N == 2:
            label = bugLabel
            title = bugTitle

        #hist(angVel, bins=bins, histtype='stepfilled', alpha=0.3, density=True, label=label, color=color[count])

        #hist(angVel, bins=bins, histtype='step', density=True, color=color[count])
    
        d[label] = angVel[:1000] / 5

        count += 1

    df = pd.DataFrame(d)

    sns.kdeplot(data=df, fill=False, common_norm=False, palette="bright",
                alpha=.5, linewidth=1, cumulative=False)
    plt.title(title)
    plt.xlim(-5, 5)
    plt.xlabel(r'$\omega$ [rad/s]')
    plt.ylabel(r'density')
    plt.show()

def plot_angular_velocity_density_distribution(angularVelocity):


    nbrExperiments = len(angularVelocity)
    nameExperiments = list(angularVelocity.keys())


    for experiment in nameExperiments:


        angVel = angularVelocity[experiment][3]
        angVel = angVel[np.nonzero(angVel)]
        #angVel = angVel[angVel >= 0.00001]
        w = int(angularVelocity[experiment][0])
        c = int(angularVelocity[experiment][1])
        b = int(angularVelocity[experiment][2])

        weightTitle = r'$N_{passive}=%d$, $N_{active}=%d$' % (c, b)
        weightLabel = r'$m_{passive}=%d\cdot10^{-3}$kg' % (w * 5 + 2)

        bugTitle = r'$N_{passive}=%d$, $m_{passive}=%d\cdot10^{-3}$kg' % (c, w * 5 + 2)
        bugLabel = r'$N_{active}=%d$' % (b)

        label = weightLabel
        title = weightTitle


        sns.distplot(angVel,
                     hist=False,
                     kde=True,
                     kde_kws={'shade': True, 'linewidth': 1},
                     label=label)

    #plt.yticks(np.linspace(0, 13000, 5), np.linspace(0, 1, 5))
    plt.axvline(x =0, color="black", alpha=1, lw=1.5, linestyle='--')
    plt.title(title)
    plt.xlim(-25, 25)
    #plt.ylim(0, 35000)
    plt.xlabel(r'angular velocity [rad/s]')
    plt.ylabel(r'density')
    plt.legend(facecolor = 'white', framealpha = 1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


listOfFiles = glob.glob('C:/Users/THOMAS/Desktop/masters_thesis_2021/main_data/*W200C10B*')
#listOfFiles = glob.glob('C:/Users/THOMAS/Desktop/master_thesis_2020/main_data_orientation/*')
listOfFiles = natsorted(listOfFiles, reverse=False)
angularVelocity = read_data(listOfFiles)

plot_angularvelocity_histogram(angularVelocity, 0)
#plot_angular_velocity_density_distribution(angularVelocity)