import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trackpy as tp
import glob
from natsort import natsorted
import ntpath
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.pyplot import quiver
import pims
import trackpy as tp
import os
from matplotlib.colors import ListedColormap
import matplotlib.colors


def plot_trajs(dataPath):
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel


    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)

    #fig, ax = plt.subplots(figsize=(1.266, 1.075), dpi=1000)
    fig, ax = plt.subplots()
    ax.set_axis_off()

    for ifile, file in enumerate(listOfFiles):
        fileName = ntpath.basename(file)
        imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/' \
                    'image_sequences/' + fileName.split('.')[0] + '/*png'

        frames = gray(pims.ImageSequence(imagePath))

        with tp.PandasHDFStore(file) as s:
            trajs = pd.concat(iter(s))
        s.close()

        trajs1 = tp.filter_stubs(trajs, 0)
        trajs2 = trajs1[(trajs1['mass'] > 4500)]

        f = 1000

        trajs3 = trajs2[trajs2.frame < f]


        tp.plot_traj(trajs3,
                     superimpose=frames[f],
                     ax=ax,
                     plot_style={'color': 'red', 'alpha': 0.6, 'linewidth': 0.5}
                     )

        outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/traj_images/' \
                  + fileName.split('.')[0] + '_traj.png'

    plt.savefig(outpath, bbox_inches='tight', dpi=1000, pad_inches=0.0)
    plt.show()

def msd_individual(dataPath):
    mpp = 701.37
    fps = 30


    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)

    for ifile, file in enumerate(listOfFiles):
        fileName = ntpath.basename(file)
        print(fileName)

        with tp.PandasHDFStore(file) as s:
            trajs = pd.concat(iter(s))

        s.close()


    im = tp.imsd(trajs, mpp, fps)

    fig, ax = plt.subplots()
    ax.plot(im.index, im, 'k-', alpha=0.6)  # black lines, semitransparent
    ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [m$^2$]',
           xlabel='lag time $t$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()

def msd_ensemble(dataPath, N):
    mpp = 0.89/1266
    fps = 30            #yeah fps = 1
    max_lagtime = 3

    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)

    #fig, ax = plt.subplots()
    plt.figure()
    for ifile, file in enumerate(listOfFiles[:]): #[listOfFiles[0], listOfFiles[3], listOfFiles[5]]
        fileName = ntpath.basename(file)
        expName = fileName.split('.')[0]
        label = expName

        nbrWeight = int(expName.split('W')[0])
        nbrObstacles = int(expName.split('W')[1].split('C')[0])
        nbrActive = int(expName.split('W')[1].split('C')[1].split('B')[0])

        if nbrObstacles == 0:
            bugTitle = r'$N_{passive}=%d$' % (nbrObstacles)
        else:
            bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg' % (nbrObstacles, nbrWeight * 5 + 2)

        bugLabel = r'$N_{active}=%d$' % (nbrActive)

        weightTitle = r'$N_{passive}=%d$ and $N_{active}=%d$' % (nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg' % (nbrWeight * 5 + 2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$' % (nbrWeight * 5 + 2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$' % (nbrObstacles)

        if N == 0:
            title = weightTitle
            label = weightLabel

        elif N == 1:
            title = obstaclesTitle
            label = obstaclesLabel

        elif N == 2:
            title = bugTitle
            label = bugLabel

        with tp.PandasHDFStore(file) as s:
            trajs = pd.concat(iter(s))

        s.close()

        trajs1 = tp.filter_stubs(trajs, 10)
        trajs2 = trajs1[(trajs1['mass'] > 4500)]
        d = tp.compute_drift(trajs2)
        trajs3 = tp.subtract_drift(trajs2.copy(), d)

        em = tp.emsd(trajs3, mpp, fps)
        p = plt.plot(em.index, em, 'o', label=label)

        n = tp.utils.fit_powerlaw(em, plot=False).n.msd
        A = tp.utils.fit_powerlaw(em, plot=False).A.msd
        #plt.text(em.index[-1]+0.2, A*em.index[-1]**n, r'$\alpha$=%.2f' % (round(n, 2)), fontsize=15)
        #plt.plot(em.index, A*em.index**n, p[0].get_color())
        print(n)

    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel(r'$\tau$ [s]')
    plt.ylabel(r'MSD($\tau$) [m$^2$]')
    plt.xlim([0, 10])
    plt.legend(loc='upper left')
    plt.show()

def velocity_calc_save(dataPath):
    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)

    outputPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/velocity_data/'
    for ifile, file in enumerate(listOfFiles[:]):
        fileName = ntpath.basename(file)
        expName = fileName.split('.')[0]
        label = expName
        outputPathData = outputPath + expName + '.h5'


        nbrWeight = int(expName.split('W')[0])
        nbrObstacles = int(expName.split('W')[1].split('C')[0])
        nbrActive = int(expName.split('W')[1].split('C')[1].split('B')[0])

        if nbrObstacles == 0:
            bugTitle = r'$N_{passive}=%d$' % (nbrObstacles)
        else:
            bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg' % (nbrObstacles, nbrWeight * 5 + 2)

        bugLabel = r'$N_{active}=%d$' % (nbrActive)

        weightTitle = r'$N_{passive}=%d$ and $N_{active}=%d$' % (nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg' % (nbrWeight * 5 + 2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$' % (nbrWeight * 5 + 2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$' % (nbrObstacles)

        with tp.PandasHDFStore(file) as s:
            trajs = pd.concat(iter(s))

        s.close()

        trajs1 = tp.filter_stubs(trajs, 10)
        trajs2 = trajs1[(trajs1['mass'] > 4500)]

        col_names = ['dx', 'dy', 'x', 'y', 'frame', 'particle']
        # Creating an empty dataframe to store results
        data = pd.DataFrame(np.zeros(shape=(1, 6), dtype=np.int64), columns=col_names)

        for item in set(trajs2.particle):
            sub = trajs2[trajs2.particle == item]

            if sub.shape[0] <= 2:
                # Cases in which particle only has 1 or 2 rows of data
                pass
            else:
                # print('Deriving velocities for particle:', str(item))
                dvx = pd.DataFrame(np.gradient(sub.x), columns=['dx', ])
                dvy = pd.DataFrame(np.gradient(sub.y), columns=['dy', ])

                new_df = pd.concat((dvx, dvy, sub.x.reset_index(drop=True), sub.y.reset_index(drop=True),
                                    sub.frame.reset_index(drop=True), sub.particle.reset_index(drop=True)),
                                   axis=1, names=col_names, sort=False)
                data = pd.concat((data, new_df), axis=0)

        # This is to get rid of the first 'np.zeros' row and to reset indexes
        data = data.reset_index(drop=True)
        data = data.drop(0)
        data = data.reset_index(drop=True)

        data.to_hdf(outputPathData, key='df', mode='w')

def plot_velocity(velocityPath, N):
    listOfFiles = glob.glob(velocityPath)
    listOfFiles = natsorted(listOfFiles)

    mpp = 0.89 / 1266
    fps = 30
    title = ''

    d = {}

    for ifile, file in enumerate(listOfFiles[::2]):
        fileName = ntpath.basename(file)
        expName = fileName.split('.')[0]
        label = expName

        nbrWeight = int(expName.split('W')[0])
        nbrObstacles = int(expName.split('W')[1].split('C')[0])
        nbrActive = int(expName.split('W')[1].split('C')[1].split('B')[0])

        if nbrObstacles == 0:
            bugTitle = r'$N_{passive}=%d$' % (nbrObstacles)
        else:
            bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg' % (nbrObstacles, nbrWeight * 5 + 2)

        bugLabel = r'$N_{active}=%d$' % (nbrActive)

        weightTitle = r'$N_{passive}=%d$ and $N_{active}=%d$' % (nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg' % (nbrWeight * 5 + 2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$' % (nbrWeight * 5 + 2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$' % (nbrObstacles)

        if N == 0:
            title = weightTitle
            label = weightLabel

        elif N == 1:
            title = obstaclesTitle
            label = obstaclesLabel

        elif N == 2:
            title = bugTitle
            label = bugLabel


        data = pd.read_hdf(file, 'df')

        # Figure out how gradient is calculated and whether or not it is velocity
        #also angle between two vectors
        # try to remove drift when plotting MSD
        # github


        vel = np.sqrt(data.dx**2 + data.dy**2)*mpp*fps
        vel = vel.values.tolist()
        print(len(vel))
        d[label] = vel[:1500]



    df = pd.DataFrame(d)
    sns.kdeplot(data=df, fill=True, common_norm=True, palette="tab10",
                alpha=.5, linewidth=0.2, cumulative=False, cut=0)
    plt.xlabel(r'v [m/s]')
    plt.xlim([0, 0.8])
    plt.title(title)
    plt.show()

def plot_orientation(velocityPath, N):
    listOfFiles = glob.glob(velocityPath)
    listOfFiles = natsorted(listOfFiles)

    mpp = 0.89 / 1266
    fps = 30
    title = ''
    rad2deg = 180/np.pi

    d = {}

    for ifile, file in enumerate(listOfFiles[::3]):
        fileName = ntpath.basename(file)
        expName = fileName.split('.')[0]
        label = expName

        nbrWeight = int(expName.split('W')[0])
        nbrObstacles = int(expName.split('W')[1].split('C')[0])
        nbrActive = int(expName.split('W')[1].split('C')[1].split('B')[0])

        if nbrObstacles == 0:
            bugTitle = r'$N_{passive}=%d$' % (nbrObstacles)
        else:
            bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg' % (nbrObstacles, nbrWeight * 5 + 2)

        bugLabel = r'$N_{active}=%d$' % (nbrActive)

        weightTitle = r'$N_{passive}=%d$ and $N_{active}=%d$' % (nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg' % (nbrWeight * 5 + 2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$' % (nbrWeight * 5 + 2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$' % (nbrObstacles)

        if N == 0:
            title = weightTitle
            label = weightLabel

        elif N == 1:
            title = obstaclesTitle
            label = obstaclesLabel

        elif N == 2:
            title = bugTitle
            label = bugLabel

        data = pd.read_hdf(file, 'df')

        # Figure out how gradient is calculated and whether or not it is velocity
        # also angle between two vectors
        # try to remove drift when plotting MSD
        # github

        theta_rad = []
        for particle in set(data.particle):
            theta_rad.extend(np.arctan2((-data.dy.values).tolist(), data.dx.values.tolist()).tolist())

        #i = 52
        #a = data[data.frame == i]
        #rawframes = pims.open(os.path.join('C:/Users/THOMAS/Desktop/master_thesis_2020/'
        #                                   'code/image_sequences/', expName, '*.png'))
        #plt.imshow(rawframes[i])
        #plt.quiver(a.x, a.y, a.dx, -a.dy, pivot='middle', headwidth=4, headlength=6, color='red')
        #plt.axis('off')
        #lt.tight_layout()
        #print(np.mod(theta_rad[i]*rad2deg+360, 360))

        dtheta_rad = np.gradient(theta_rad)
        #dtheta_rad = [y - x for x, y in zip(theta_rad, theta_rad[1:])]
        #print(np.min(dtheta_rad), np.max(dtheta_rad))

        #dtheta_rad[np.abs(dtheta_rad) > (np.pi/2)] = 0
        #dtheta_rad = dtheta_rad[np.abs(dtheta_rad) < (np.pi/2)]
        #dtheta_rad = np.mod(dtheta_rad, 1.4)*np.sign(dtheta_rad)


        d[label] = dtheta_rad[:100000]*fps

    df = pd.DataFrame(d)
    sns.kdeplot(data=df, fill=False, common_norm=False, palette="bright",
                alpha=.5, linewidth=1, cumulative=False)



    plt.xlabel(r'$d\theta/dt$ [rad/s]')
    #plt.xlim([0, 1])
    plt.title(title)
    plt.show()

def plot_mean_velocity(velocityPath):
    listOfFiles = glob.glob(velocityPath)
    listOfFiles = natsorted(listOfFiles)

    mpp = 0.89 / 1266
    fps = 30
    title = ''

    d = {'W': [], 'C': [], 'B': [], 'V': []}

    for ifile, file in enumerate(listOfFiles[:]):
        fileName = ntpath.basename(file)
        expName = fileName.split('.')[0]


        nbrWeight = int(expName.split('W')[0])
        nbrObstacles = int(expName.split('W')[1].split('C')[0])
        nbrActive = int(expName.split('W')[1].split('C')[1].split('B')[0])

        if nbrObstacles == 0:
            bugTitle = r'$N_{passive}=%d$' % (nbrObstacles)
        else:
            bugTitle = r'$N_{passive}=%d$ and $m_{passive}=%d\cdot10^{-3}$kg' % (nbrObstacles, nbrWeight * 5 + 2)

        bugLabel = r'$N_{active}=%d$' % (nbrActive)

        weightTitle = r'$N_{passive}=%d$ and $N_{active}=%d$' % (nbrObstacles, nbrActive)
        weightLabel = r'$m_{passive}$=%d$\cdot10^{-3}$kg' % (nbrWeight * 5 + 2)

        obstaclesTitle = r'$m_{passive}=%d\cdot10^{-3}$kg and $N_{active}=%d$' % (nbrWeight * 5 + 2, nbrActive)
        obstaclesLabel = r'$N_{passive}=%d$' % (nbrObstacles)

        data = pd.read_hdf(file, 'df')

        # Figure out how gradient is calculated and whether or not it is velocity
        # also angle between two vectors
        # try to remove drift when plotting MSD
        # github

        vel = np.sqrt(data.dx ** 2 + data.dy ** 2) * mpp * fps
        vel = vel.values.tolist()

        d['W'].append(nbrWeight)
        d['C'].append(nbrObstacles)
        d['B'].append(nbrActive)
        d['V'].append(np.mean(vel))


    df = pd.DataFrame(d)


    g = sns.lineplot(data=df, x='C', y='V', hue='W', palette='bright')
    plt.xlabel(r'$N_{passive}$')
    plt.xlim([100, 1300])
    plt.ylabel(r'$\langle v\rangle$ [m/s]')
    plt.title('Mean velocity active particles')
    g.legend(title=r'$m_{passive}$', labels=['2$\cdot10^{-3}$kg', '7$\cdot10^{-3}$kg', '12$\cdot10^{-3}$kg',
                                             '17$\cdot10^{-3}$kg'])
    plt.show()


dataPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data/0W1100C15B*'
#velocityPath = 'C:/Users/THOMAS/Desktop/master_thesis_2020/code/velocity_data/*W700C20B*'
plot_trajs(dataPath)
#msd_individual(dataPath)
#msd_ensemble(dataPath, 0)
#velocity_calc_save(velocityPath)
#plot_velocity(velocityPath, 1)
#plot_orientation(velocityPath, 1) # doesn't work
#plot_mean_velocity(velocityPath)
