%% Unit_test_2
% code to analyse frozen frames
clc
clear variables


x = [0 1 2 3 0 5 6 7 8 0 10];
zpos = find(~[0 x 0]);
[~, grpidx] = max(diff(zpos));
y = x(zpos(grpidx):zpos(grpidx+1)-2);
ind = [zpos(grpidx):zpos(grpidx+1)-2];

x(x<1) = 1