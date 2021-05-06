%%
close all
clear all
names = {'0W0C16B','0W300C15B','1W300C15B','2W300C15B',...
    '0W700C15B','1W700C15B','2W700C15B','1W1000C15B'};
    cellLegend = {};
for ii=1:numel(names)
    folder = names{ii};
load(strcat('..\Positions\New_Conditions_2\',names{ii},'\position.mat'))

N = numel(X(:,1));
t=(1:N)/25;
nBugs=15;
%k=1;

%Mean squared displacement
msdx = zeros(1,N-1);
msdy = zeros(1,N-1);
    
for k=1:nBugs
    x = transpose(X(1:N,k));
    y = transpose(Y(1:N,k));


    for n = 1:N-1
        msdx(n) = msdx(n) + sum( (x(1+n:N) - x(1:N-n)).^2 )/(N-n);
        msdy(n) = msdy(n) + sum( (y(1+n:N) - y(1:N-n)).^2 )/(N-n);
    end
end
msdx=msdx/nBugs;
msdy=msdy/nBugs;

msd = (msdx+msdy)/2;

%figure(1)
%loglog(t(1:N-1), msdx,t(1:N-1), msdy)
plot(1:N-1, msd)
cellLegend=[cellLegend,names{ii}];
legend(cellLegend)

hold on
end
xlabel('time(s)')
ylabel('msd(pixels^2)')
hold off
