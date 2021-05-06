%% unit test 2

clc
clear varibles

x = [1 2 3 4 5];
y = [1 2 3 4 5];

m = msd(x, y);
tau = linspace(1, length(m), length(m));
%plot(tau, m)

s = [2, 0, 4, 0, 1];

out=nnz(~s)
