% Series of examples to demonstrate the use of ParticleCylindrical.
%
% See also Particle, ParticleCylindrical.

%   Author: Giovanni Volpe
%   Revision: 1.0.0
%   Date: 2015/01/01

example('Use of ParticleCylindrical')

%% DEFINITION OF PARTICLECYLINDRICAL
exampletitle('DEFINITION OF PARTICLECYLINDRICAL')

examplecode('v = Vector(2,2,2,1,1,1);')
examplecode('r = 1;')
examplecode('nm = 1;')
examplecode('np = 1.5;')
examplecode('bead = ParticleCylindrical(v,r,nm,np);')
examplewait()

%% PLOTTING OF PARTICLECYLINDRICAL
exampletitle('PLOTTING OF PARTICLECYLINDRICAL')

figure
title('PARTICLECYLINDRICAL')
hold on
axis equal
grid on
view(3)
xlabel('x')
ylabel('y')
zlabel('z')

examplecode('bead.plot();')
examplewait()

%% SCATTERING
exampletitle('SCATTERING')

examplecode('mr = 3;')
examplecode('nr = 2;')
examplecode('v = Vector(zeros(mr,nr),zeros(mr,nr),zeros(mr,nr),rand(mr,nr),rand(mr,nr),rand(mr,nr));')
examplecode('P = ones(mr,nr);')
examplecode('pol = Vector(zeros(mr,nr),zeros(mr,nr),zeros(mr,nr),ones(mr,nr),ones(mr,nr),ones(mr,nr)); pol = v*pol;')
examplecode('r = Ray(v,P,pol);')
examplewait()

examplecode('r.plot(''color'',''k'');')
examplewait()

examplecode('r_vec = bead.scattering(r)')
examplewait()

examplecode('rr = r_vec(1).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(1).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(2).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(2).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(3).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(3).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(4).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(4).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

examplecode('rr = r_vec(5).r;')
examplecode('rr.plot(''color'',''r'');')
examplecode('rt = r_vec(5).t;')
examplecode('rt.plot(''color'',''b'');')
examplewait()

%% FORCE
exampletitle('FORCE')

examplecode('F = bead.force(r)*1e+15 % fN')
examplewait()

%% TORQUE
exampletitle('TORQUE')

examplecode('T = bead.torque(r,1e-21)*1e+21 % fN*nm')