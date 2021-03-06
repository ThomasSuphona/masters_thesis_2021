% Series of examples to demonstrate the use of OTCalibPSD.
%
% See also OTCalib, OTCalibPSD.

%   Author: Giovanni Volpe
%   Revision: 1.0.0  
%   Date: 2015/01/01

clear all; close all; clc;

load('1D.mat')

Sx = 1;
otc = OTCalibPSD(Vx,Sx,dt,R,eta,T);

kBT = otc.kBT
D = otc.D
gamma = otc.gamma
number_of_samples = otc.samples()
number_of_windows = otc.windows()

otc = otc.calibrate( ...
    'fmin',1, ...
    'fmax',1e+4, ...
    'blocking','lin', ...
    'binsnumber', 500, ...
    'verbose',true, ...
    'displayon',true ...
    );

otc = otc.set_au2m(otc.Sx_fit);

otc = otc.calibrate( ...
    'fmin',1, ...
    'fmax',1e+4, ...
    'blocking','lin', ...
    'binsnumber', 500, ...
    'verbose',true, ...
    'displayon',true ...
    );

otc.plottraj()
otc.plotcalib()
otc.printcalib()