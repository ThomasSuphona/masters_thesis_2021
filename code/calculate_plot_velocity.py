import numpy as np
import glob
import scipy.io as scio
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from astropy.visualization import hist
from natsort import natsorted
from scipy.signal import savgol_filter
import seaborn as sns
from sklearn.neighbors import KernelDensity


sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})




def calculate_velocity(listOfFiles):

    velocities = {}
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
        pTrajX = passiveTrajectoriesX_px[passiveIndices, :].toarray()*pixelsToMeterPassive
        pTrajY = passiveTrajectoriesY_px[passiveIndices, :].toarray()*pixelsToMeterPassive

        aTrajX = activeTrajectoriesX_px[activeIndices, :].toarray()*pixelsToMeterActive
        aTrajY = activeTrajectoriesY_px[activeIndices, :].toarray()*pixelsToMeterActive


        activeVelocityPoints = np.count_nonzero(aTrajX)-aTrajX.shape[0]
        activeVelocity = np.zeros(activeVelocityPoints)

        lower = 0
        upper = 0


        for iActiveParticle in range(nbrActiveUse):
            activeTrajNonzeroX_i = aTrajX[iActiveParticle, np.nonzero(aTrajX[iActiveParticle, :])][0]
            activeTrajNonzeroY_i = aTrajY[iActiveParticle, np.nonzero(aTrajY[iActiveParticle, :])][0]

            dxActive_i = np.asarray([(l - m) for (l, m) in zip(activeTrajNonzeroX_i[1:], activeTrajNonzeroX_i)])
            dyActive_i = np.asarray([(l - m) for (l, m) in zip(activeTrajNonzeroY_i[1:], activeTrajNonzeroY_i)])

            upper = len(dxActive_i) + lower

            activeVelocity[lower:upper] = np.sqrt(dxActive_i**2 + dyActive_i**2)*frameRate_Hz**-1

            lower = upper

        velocities[nameExperiment] = (nbrWeights, nbrPassiveTrue, nbrActiveTrue, activeVelocity)

    return velocities

def plot_velocities1(velocities):
    nbrExperiments = len(velocities)
    #velocities = OrderedDict(velocities)
    nameExperiments = list(velocities.keys())

    # Create the data
    nbrVelPoints = np.sum([len(velocities[experiment][3]) for experiment in nameExperiments])

    g = [''] * nbrVelPoints
    x = np.zeros(nbrVelPoints)
    upper = 0
    lower = 0



    for i in range(nbrExperiments):
        varyNbrBugsLabel = f'{velocities[nameExperiments[i]][2]}B'
        varyNbrPassiveLabel = f'{velocities[nameExperiments[i]][1]}C'
        varyNbrWeightLabel = f'{velocities[nameExperiments[i]][0]}W'

        nbrPoints = len(velocities[nameExperiments[i]][3])
        upper = nbrPoints + lower

        g[lower:upper] = np.tile(varyNbrWeightLabel, nbrPoints)

        x[lower:upper] = velocities[nameExperiments[i]][3]

        lower = upper

    g = np.asarray(g)

    df = pd.DataFrame(dict(x=x, g=g))

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "x")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()

    #for iExperiment in range(nbrExperiments):
    #    nbrWeights = velocities[nameExperiments[iExperiment]][0]
    #    nbrPassive = velocities[nameExperiments[iExperiment]][1]
    #    nbrActive = velocities[nameExperiments[iExperiment]][2]
    #    velocity = velocities[nameExperiments[iExperiment]][3]
    #    nbrVelPoints = len(velocity)

def plot_velocities2(velocities):


    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())


    bins = 100

    x1 = velocities[nameExperiments[0]][3]
    x2 = velocities[nameExperiments[1]][3]
    x3 = velocities[nameExperiments[2]][3]
    x4 = velocities[nameExperiments[3]][3]

    w1 = int(velocities[nameExperiments[0]][0])
    w2 = int(velocities[nameExperiments[1]][0])
    w3 = int(velocities[nameExperiments[2]][0])
    w4 = int(velocities[nameExperiments[3]][0])

    c = int(velocities[nameExperiments[0]][1])
    b = int(velocities[nameExperiments[0]][2])
    labels = [f'{w1*5+2}g', f'{w2*5+2}g', f'{w3*5+2}g', f'{w4*5+2}g']
    colors = ['red', 'black', 'lime', 'blue']

    plt.hist([x1, x2, x3, x4], bins=bins, density=True,
             color=colors, label=labels, histtype='bar')

    #for experiment in nameExperiments:
        #velocity = velocities[experiment][3]
        #w = int(velocities[experiment][0])
        #c = int(velocities[experiment][1])
        #b = int(velocities[experiment][2])

        #plt.hist(velocity,
        #         bins=bins,
        #         alpha=0.6,
        #         density=True,
        #         label=f'{w*5+2}g'
        #         )

        #sns.distplot(velocity,
        #             hist=False,
        #             kde=True,
        #             norm_hist=True,
        #             bins=bins,
                     #fit=norm,
                     #color='darkblue',
                     #hist_kws={'edgecolor': 'black'},
        #             kde_kws={'shade': True, 'linewidth': 1},
        #             label=f'{w*5+2}g'
        #             )

    plt.title(f'Velocity histogram of {b} active particles with {c} passive part.')
    plt.xlim(0, 0.0002)
    plt.xlabel(r'velocity [m/s]')
    plt.legend(facecolor='white', framealpha=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()

def plot_velocities3(path):
    # as function of number of passive
    velMat = np.loadtxt(path)
    nbrActive = 15

    #for i in range(4):
    #    y = velMat[np.argwhere((velMat[:, 0]==i) & (velMat[:, 2]==25))][:, 0, 3]
    #    x = velMat[np.argwhere((velMat[:, 0]==i) & (velMat[:, 2]==25))][:, 0, 1]

    #   y = y[np.argwhere(x >= 700)]
    #    x = x[np.argwhere(x >= 700)]

    #    plt.plot(x, y, '*', label=f'{i*5+2}g')

    x_0W = velMat[np.argwhere((velMat[:, 0]==0) & (velMat[:, 2]==nbrActive))][:, 0, 1]
    y_0W = velMat[np.argwhere((velMat[:, 0]==0) & (velMat[:, 2]==nbrActive))][:, 0, 3]
    idx_sort_0W = np.argsort(x_0W)
    x_0W = x_0W[idx_sort_0W]
    y_0W = y_0W[idx_sort_0W]

    x_1W = velMat[np.argwhere((velMat[:, 0] == 1) & (velMat[:, 2] == nbrActive))][:, 0, 1]
    y_1W = velMat[np.argwhere((velMat[:, 0] == 1) & (velMat[:, 2] == nbrActive))][:, 0, 3]
    idx_sort_1W = np.argsort(x_1W)
    x_1W = x_1W[idx_sort_1W]
    y_1W = y_1W[idx_sort_1W]

    x_2W = velMat[np.argwhere((velMat[:, 0] == 2) & (velMat[:, 2] == nbrActive))][:, 0, 1]
    y_2W = velMat[np.argwhere((velMat[:, 0] == 2) & (velMat[:, 2] == nbrActive))][:, 0, 3]
    idx_sort_2W = np.argsort(x_2W)
    x_2W = x_2W[idx_sort_2W]
    y_2W = y_2W[idx_sort_2W]

    x_3W = velMat[np.argwhere((velMat[:, 0] == 3) & (velMat[:, 2] == 25))][:, 0, 1]
    y_3W = velMat[np.argwhere((velMat[:, 0] == 3) & (velMat[:, 2] == 25))][:, 0, 3]
    idx_sort_3W = np.argsort(x_3W)
    x_3W = x_3W[idx_sort_3W]
    y_3W = y_3W[idx_sort_3W]



    plt.plot(x_0W, y_0W, '-', label=r'$m_{passive}=2\cdot10^{-3}$kg')
    plt.plot(x_1W, y_1W, ':', label=r'$m_{passive}=7\cdot10^{-3}$kg')
    plt.plot(x_2W, y_2W, '-.', label=r'$m_{passive}=12\cdot10^{-3}$kg')
    plt.plot(x_3W, y_3W, '--', label=r'$m_{passive}=17\cdot10^{-3}$kg')


    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title(r'$N_{active}=%d$'%nbrActive)
    plt.legend(facecolor='white', framealpha=1)
    plt.xlabel(r'$N_{passive}$')
    plt.ylabel(r'$<v_{active}>$ [m/s]')
    plt.show()

def plot_velocities4(path):
    # As function of number of bugs
    velMat = np.loadtxt(path)
    #print(len(velMat[np.argwhere((velMat[:, 0]==2) & (velMat[:, 2]==25) & (velMat[:, 1]==1300))]))
    c = 100
    for i in range(4):
        y = velMat[np.argwhere((velMat[:, 0]==i) & (velMat[:, 1]==c))][:, 0, 3]
        x = velMat[np.argwhere((velMat[:, 0]==i) & (velMat[:, 1]==c))][:, 0, 2]


        plt.plot(x, y, '*', label=f'{i*5+2}g')

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title('Mean velocity of active particles')
    plt.legend(facecolor='white', framealpha=1)
    plt.xlabel(r'number active particles')
    plt.ylabel(r'mean velocity [m/s]')
    plt.show()

def calculate_mean_velocity(velocities):
    savePath = 'F:/Thomas_Suphona/master_thesis/version3/mean_velocity_data/mean_velocities.txt'

    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())

    velMat = np.zeros((nbrExperiments, 4))

    for iExperiment, experiment in enumerate(nameExperiments):
        velMat[iExperiment, 0] = velocities[experiment][0]
        velMat[iExperiment, 1] = velocities[experiment][1]
        velMat[iExperiment, 2] = velocities[experiment][2]
        velMat[iExperiment, 3] = np.nanmean(velocities[experiment][3])


    np.savetxt(savePath, velMat)

def plot_velocities5(velocities):


    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())

    bins = 100


    for experiment in nameExperiments:
        velocity = velocities[experiment][3]
        velocity = velocity[np.nonzero(velocity)]
        w = int(velocities[experiment][0])
        c = int(velocities[experiment][1])
        b = int(velocities[experiment][2])
        #print(len(np.argwhere(velocity == np.min(velocity))), w)

        hist(velocity, bins='freedman', histtype='stepfilled',
             alpha=0.5, density=True, label=f'{w*5+2}g')


    plt.title(f'Velocity histogram of {b} active particles \n with {c} passive particles')
    #plt.title(f'Velocity histogram of active particles')
    plt.xlim(0, 0.0004)
    plt.xlabel(r'velocity [m/s]')
    plt.ylabel(r'count')
    plt.legend(facecolor='white', framealpha=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()

def plot_velocities6(velocities):


    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())

    bins = 100

    color = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    count = 0

    for experiment in nameExperiments:
        velocity = velocities[experiment][3]
        velocity = velocity[np.nonzero(velocity)]
        #velocity = velocity[velocity >= 0.00001]
        w = int(velocities[experiment][0])
        c = int(velocities[experiment][1])
        b = int(velocities[experiment][2])
        #print(len(np.argwhere(velocity == np.min(velocity))), w)

        weightTitle = r'$N_{passive}=%d$, $N_{active}=%d$'%(c, b)
        weightLabel = r'$m_{passive}=%d\cdot10^{-3}$kg'%(w*5+2)

        bugTitle = r'$N_{passive}=%d$, $m_{passive}=%d\cdot10^{-3}$kg' %(c, w*5+2)
        bugLabel = r'$N_{active}=%d$' % (b)

        hist(velocity,
             bins='freedman',
             histtype='stepfilled',
             alpha=0.3,
             density=True,
             label=weightLabel,
             color=color[count]
             )

        hist(velocity,
             bins='freedman',
             histtype='step',
             density=True,
             color=color[count]
             )

        count = count + 1

    #plt.yticks(np.linspace(0, 55000, 5), np.linspace(0, 1, 5))
    plt.title(weightTitle)
    plt.xlim(0, 0.0004)
    #plt.ylim(0, 20000)
    plt.xlabel(r'velocity [m/s]')
    plt.ylabel(r'density')
    plt.legend(facecolor='white', framealpha=1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()

def plot_velocities_time(velocities):


    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())
    nameExperiment = nameExperiments[0]

    velocity = velocities[nameExperiment][3]
    velocity = velocity[np.nonzero(velocity)]
    #velocity = velocity[velocity >= 0.00001]
    nbrFrames = len(velocity)
    frameRate = 30.00003
    duration_s = nbrFrames/frameRate

    time_vec = np.linspace(0, duration_s, nbrFrames, endpoint=False)

    w = int(velocities[nameExperiment][0])
    c = int(velocities[nameExperiment][1])
    b = int(velocities[nameExperiment][2])

    velocity_hat = savgol_filter(velocity, 1001, 4)

    plt.plot(time_vec, velocity_hat, '-b', label=r'$m_{passive}=17\cdot10^{-3}$kg')
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.title('Velocity of one active particle')
    #plt.legend(facecolor='white', framealpha=1)
    plt.xlabel(r'$t$ [s]')
    plt.ylabel(r'$v_{active}$ [m/s]')

    plt.text(28, 0.24E-4, r'(a)', fontsize=15)
    plt.text(60, 1.75E-4, r'(b)', fontsize=15)
    plt.text(108, 1.90E-4, r'(c)', fontsize=15)
    plt.text(142, 0.15E-4, r'(d)', fontsize=15)

    plt.show()

    #plt.plot(velocity, '--', label=r'$m_{passive}=17\cdot10^{-3}$kg')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.title(r'$N_{active}=%d$' % nbrActive)
    #plt.legend(facecolor='white', framealpha=1)
    #plt.xlabel(r'$N_{passive}$')
    #plt.ylabel(r'$<v_{active}>$ [m/s]')
    #plt.show()


# need improvement
def plot_velocities_density_distribution(velocities):


    nbrExperiments = len(velocities)
    nameExperiments = list(velocities.keys())


    for experiment in nameExperiments:


        velocity = velocities[experiment][3]
        velocity = velocity[np.nonzero(velocity)]
        #velocity = velocity[velocity >= 0.00001]
        w = int(velocities[experiment][0])
        c = int(velocities[experiment][1])
        b = int(velocities[experiment][2])

        weightTitle = r'$N_{passive}=%d$, $N_{active}=%d$' % (c, b)
        weightLabel = r'$m_{passive}=%d\cdot10^{-3}$kg' % (w * 5 + 2)

        bugTitle = r'$N_{passive}=%d$, $m_{passive}=%d\cdot10^{-3}$kg' % (c, w * 5 + 2)
        bugLabel = r'$N_{active}=%d$' % (b)

        label = bugLabel
        title = bugTitle


        sns.distplot(velocity,
                     hist=False,
                     kde=True,
                     kde_kws={'shade': True, 'linewidth': 1},
                     label=label)

    #plt.yticks(np.linspace(0, 13000, 5), np.linspace(0, 1, 5))
    plt.title(title)
    plt.xlim(0, 0.0005)
    plt.ylim(0, 35000)
    plt.xlabel(r'velocity [m/s]')
    plt.ylabel(r'density')
    plt.legend(facecolor = 'white', framealpha = 1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


# Need improvements
def calculate_velocity_no_obstacles(listOfFiles):  # Need improvements
    velocities = {}
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
        #passiveTrajectoriesX_px = data['Data'][0, 0][11]
        #passiveTrajectoriesY_px = data['Data'][0, 0][12]
        activeTrajectoriesX_px = data['Data'][0, 0][13]
        activeTrajectoriesY_px = data['Data'][0, 0][14]
        pixelsToMeterActive = data['Data'][0, 0][15][0][0]
        nbrActiveTrue = data['Data'][0, 0][16][0][0]
        nbrActiveTracked = data['Data'][0, 0][17][0][0]
        nbrWeights = data['Data'][0, 0][18][0][0]
        #nbrWeights_kg = data['Data'][0, 0][19][0][0]
        #passiveIndices = data['Data'][0, 0][20][:, 0] - 1
        activeIndices = data['Data'][0, 0][21][:, 0] - 1
        activeOrientation = data['Data'][0, 0][22]
        nameExperiment = f'{nbrWeights}W{nbrPassiveTrue}C{nbrActiveTrue}B'

        nbrPassiveUse = len(passiveIndices)
        nbrActiveUse = len(activeIndices)

        # Process and convert to SI units
        #pTrajX = passiveTrajectoriesX_px[passiveIndices, :].toarray() * pixelsToMeterPassive
        #pTrajY = passiveTrajectoriesY_px[passiveIndices, :].toarray() * pixelsToMeterPassive

        aTrajX = activeTrajectoriesX_px[activeIndices, :].toarray() * pixelsToMeterActive
        aTrajY = activeTrajectoriesY_px[activeIndices, :].toarray() * pixelsToMeterActive

        activeVelocityPoints = np.count_nonzero(aTrajX) - aTrajX.shape[0]
        activeVelocity = np.zeros(activeVelocityPoints)

        lower = 0
        upper = 0

        for iActiveParticle in range(nbrActiveUse):
            activeTrajNonzeroX_i = aTrajX[iActiveParticle, np.nonzero(aTrajX[iActiveParticle, :])][0]
            activeTrajNonzeroY_i = aTrajY[iActiveParticle, np.nonzero(aTrajY[iActiveParticle, :])][0]

            dxActive_i = np.asarray([(l - m) for (l, m) in zip(activeTrajNonzeroX_i[1:], activeTrajNonzeroX_i)])
            dyActive_i = np.asarray([(l - m) for (l, m) in zip(activeTrajNonzeroY_i[1:], activeTrajNonzeroY_i)])

            upper = len(dxActive_i) + lower

            activeVelocity[lower:upper] = np.sqrt(dxActive_i ** 2 + dyActive_i ** 2) * frameRate_Hz ** -1

            lower = upper

        velocities[nameExperiment] = (nbrWeights, nbrPassiveTrue, nbrActiveTrue, activeVelocity)

    return velocities




listOfFiles = glob.glob('C:/Users/THOMAS/Desktop/master_thesis_2020/main_data/0W100C*')
#listOfFiles = glob.glob('E:/Thomas_Suphona/master_thesis/version3/no_obstacles_data/converted/0W0C21B.mat')
listOfFiles = natsorted(listOfFiles, reverse=True)
velocities = calculate_velocity(listOfFiles)
#velocities = calculate_velocity_no_obstacles(listOfFiles)


#plot_velocities5(velocities)
plot_velocities6(velocities)
#plot_velocities_time(velocities)
#calculate_mean_velocity(velocities)
#plot_velocities_density_distribution(velocities)

# All calculations connecting to mean velocities
savePath = 'E:/Thomas_Suphona/master_thesis/version3/mean_velocity_data/mean_velocities.txt'
#plot_velocities3(savePath)
#plot_velocities4(savePath)

