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
import plot_lib as pl
import matplotlib.image as mpimg
import scipy.io as scio

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

def plot_trajs(dataPath):
    @pims.pipeline
    def gray(image):
        return image[:, :, 1]  # Take just the green channel


    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)
    listOfFiles = listOfFiles[::-1]
    listOfFiles = listOfFiles[::2]
    
    pl.update_settings(usetex=True)
    fig, axs = pl.create_fig(ncols=1, nrows=4, height=1.65)
    

    for i, ax in enumerate(fig.axes):
        #ax.set_axis_off()
        ax.tick_params(top=False, bottom=False, left=False, right=False,
               labelleft=False, labelbottom=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        for ifile, file in enumerate(listOfFiles[i:i+1]):
            fileName = ntpath.basename(file)
            frameNbr = 1000
            imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/' \
                        'image_sequences/' + fileName.split('.')[0] +  \
                        '/frame-{:04d}.png'.format(frameNbr)
          
            print(fileName)
            frame = gray(pims.ImageSequence(imagePath))
            #frames = pims.ImageSequence(imagePath) # old vids

            with tp.PandasHDFStore(file) as s:
                trajs = pd.concat(iter(s))
            s.close()

            trajs = tp.filter_stubs(trajs, 0)
            trajs = trajs[(trajs['mass'] > 4500)]
            #trajs2 = trajs1[(trajs1['mass'] > 3000)] # for older vids
            
            f_last = np.max(trajs.frame)
            
  
            trajs = trajs[trajs.frame < frameNbr]
            
            
            tp.plot_traj(trajs,
    #                     superimpose=frames[f],
                         ax=ax,
                         plot_style={'color': 'red', 'alpha': 0.7, 'linewidth': 1}
                         )
            
            s = fileName.split('.')[0]
            w = int(s.split('W')[0])
            c = int(s.split('W')[1].split('C')[0])
            b = int(s.split('W')[1].split('C')[1].split('B')[0])
            clabel = '$N_{passive}=%d$'%c
            wlabel = '$m_{passive}=%d\cdot10^{-3}$kg'%(2+5*w)
            blabel = '$N_{active}=%d$'%b
            label = blabel
            #pl.add_label(ax, text=label)
            ax.set_title(label)

    outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/traj_images/1W1100CNB_traj.png'
    
    #plt.subplots_adjust(hspace = .01)
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
    #lst = ['C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data/0W800C20B.h5', 
    #'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data/0W900C20B.h5', 
    #'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data_old/C20B838P0M.h5']
    #listOfFiles = lst
    
    mpp = 0.89/1266
    fps = 30            #yeah fps = 1
    max_lagtime = 500    # frames

    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)
    listOfFiles = listOfFiles[::3]
    

    pl.update_settings(usetex=True)
    fig, axs = pl.create_fig(ncols=1, nrows=1, height=1.65)
    
    for i, ax in enumerate(fig.axes):
        for ifile, file in enumerate(listOfFiles[:]):
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
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/msd_plots/NW{:d}C{:d}B_msd.png'.format(nbrObstacles, nbrActive)
            elif N == 1:
                title = obstaclesTitle
                label = obstaclesLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/msd_plots/{:d}WNC{:d}B_msd.png'.format(nbrWeight, nbrActive)
            elif N == 2:
                title = bugTitle
                label = bugLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/msd_plots/{:d}W{:d}CNB_msd.png'.format(nbrWeight, nbrObstacles)

            with tp.PandasHDFStore(file) as s:
                trajs = pd.concat(iter(s))

            s.close()

            if 'P' in expName:
                trajs = tp.filter_stubs(trajs, 10)
                trajs = trajs[(trajs['mass'] > 2000)] # for older vids 3000
            else:
                trajs = tp.filter_stubs(trajs, 10)  #  10
                trajs = trajs[(trajs['mass'] > 4500)] # 4500

            
           
            em = tp.emsd(trajs, mpp, fps, max_lagtime=max_lagtime)
            p = ax.plot(em.index, em, 'o', label=label, markersize=5)
            #p = ax.plot(em.index, em/em.index, 'o', label=label)

            n = tp.utils.fit_powerlaw(em, plot=False).n.msd
            A = tp.utils.fit_powerlaw(em, plot=False).A.msd

        ax.set_title(title)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'lagtime $\tau$ [s]')
        ax.set_ylabel(r'MSD($\tau$) [$m^2$]')
        #ax.set_ylabel(r'MSD($\tau$)/$\tau$ [$m^2$]')

    
    
    #outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/msd_plots/msd.png'
    #plt.subplots_adjust(hspace = .01)
    #plt.savefig(outpath, bbox_inches='tight', dpi=1000, pad_inches=0.0)
    plt.legend(loc='upper left', prop={'size': 5})
    plt.savefig(outpath, bbox_inches='tight')
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
    mpp = 0.89/1266
    fps = 30            #yeah fps = 1
    max_lagtime = 500    # frames

    listOfFiles = glob.glob(velocityPath)
    listOfFiles = natsorted(listOfFiles)
    listOfFiles = listOfFiles[::2]
    
    pl.update_settings(usetex=True)
    fig, axs = pl.create_fig(ncols=1, nrows=1, height=1.65)

    d = {}
    
    for i, ax in enumerate(fig.axes):
        for ifile, file in enumerate(listOfFiles[:]):
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
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/NW{:d}C{:d}B_vel.png'.format(nbrObstacles, nbrActive)
            elif N == 1:
                title = obstaclesTitle
                label = obstaclesLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/{:d}WNC{:d}B_vel.png'.format(nbrWeight, nbrActive)
            elif N == 2:
                title = bugTitle
                label = bugLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/{:d}W{:d}CNB_vel.png'.format(nbrWeight, nbrObstacles)

            data = pd.read_hdf(file, 'df')

            vel = np.sqrt(data.dx**2 + data.dy**2)*mpp*fps
            vel = vel.values.tolist()
            print(len(vel))
            d[label] = vel[:1500]
            
        df = pd.DataFrame(d)
      
        gfg = sns.kdeplot(data=df, fill=True, common_norm=True, palette="tab10",
                alpha=.5, linewidth=0.2, cumulative=False, cut=0, ax=ax, legend=True)

        ax.set_title(title)
        ax.set_xlabel(r'v [m/s]')
        ax.set_ylabel('Density')
      
    #plt.setp(gfg.get_legend().get_texts(), fontsize='2')
    plt.xlabel(r'v [m/s]')
    plt.xlim([0, 0.8])
    plt.savefig(outpath, bbox_inches='tight')
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

def plot_images(dataPath):
   
    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)

    
    #fig, axs = plt.subplots(nrows=4, ncols=1)
    
    pl.update_settings(usetex=True)
    fig, axs = pl.create_fig(ncols=1, nrows=4, height=1.65)
    
    listOfFiles = listOfFiles[::-1]
    listOfFiles = listOfFiles[::2]
    

    for i, ax in enumerate(fig.axes):
        ax.set_axis_off()
        for ifile, file in enumerate(listOfFiles[i:i+1]):
            fileName = ntpath.basename(file)
            frameNbr = 1000
            
            imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/' \
                        'image_sequences/' + fileName.split('.')[0] +  \
                        '/frame-{:04d}.png'.format(frameNbr)
            
            
            img = mpimg.imread(imagePath)
            ax.imshow(img)

    outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/traj_images/' \
              + 'NB.png'
    plt.subplots_adjust(hspace = .01)
    plt.savefig(outpath, bbox_inches='tight', dpi=1000, pad_inches=0.0)
    plt.show()

def plot_orientation_old(dataPath, N):

    listOfFiles = glob.glob(dataPath)
    listOfFiles = natsorted(listOfFiles)
    listOfFiles = listOfFiles[::2]

    angularVelocity = read_data(listOfFiles)
    nameExperiments = list(angularVelocity.keys())

    pl.update_settings(usetex=True)
    fig, axs = pl.create_fig(ncols=1, nrows=1, height=1.65)
    
    d = {}
    
    for i, ax in enumerate(fig.axes):
        for iexp, experiment in enumerate(nameExperiments[:]):

            angVel = angularVelocity[experiment][3]
            #angVel = angVel[np.nonzero(angVel)]
            #angVel = angVel[angVel >= 0.00001]

            nbrWeight = int(angularVelocity[experiment][0])
            nbrObstacles = int(angularVelocity[experiment][1])
            nbrActive = int(angularVelocity[experiment][2])

            label = experiment


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
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/NW{:d}C{:d}B_angvel.png'.format(nbrObstacles, nbrActive)
            elif N == 1:
                title = obstaclesTitle
                label = obstaclesLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/{:d}WNC{:d}B_angvel.png'.format(nbrWeight, nbrActive)
            elif N == 2:
                title = bugTitle
                label = bugLabel
                outpath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/velocity_plots/{:d}W{:d}CNB_angvel.png'.format(nbrWeight, nbrObstacles)

            print(len(angVel))
            d[label] = angVel[:1000]/ 5
            
        df = pd.DataFrame(d)
      
        gfg = sns.kdeplot(data=df, fill=False, common_norm=False, palette="bright",
                alpha=.5, linewidth=1, cumulative=False, ax=ax, legend=True)

        ax.set_title(title)
        ax.set_xlabel(r'$\omega$ [rad/s]')
        ax.set_ylabel('Density')
        ax.set_xlim([-5, 5])
     
      
    plt.setp(gfg.get_legend().get_texts(), fontsize='4')
    plt.savefig(outpath, bbox_inches='tight')
    plt.show()

dataPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/traj_data/1W*C10B*'
#imagePath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/image_sequences/1W1000C*'
velocityPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/code/velocity_data/2W*C15B*'
angvelPath = 'C:/Users/THOMAS/Desktop/masters_thesis_2021/main_data/1W600C*'

#plot_trajs(dataPath)
#msd_individual(dataPath)
#msd_ensemble(dataPath, 1)
#velocity_calc_save(velocityPath)
#plot_orientation(velocityPath, 1) # doesn't work
#plot_mean_velocity(velocityPath)
#plot_images(imagePath)
#plot_velocity(velocityPath, 1)
plot_orientation_old(angvelPath, 2)

