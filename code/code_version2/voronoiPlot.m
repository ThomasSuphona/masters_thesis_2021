load('C:\Users\Everybody\Desktop\Quentin_Pikeroen\Quentin_Pikeroen\Code\Positions\New_Conditions_2\1W700C15B\position.mat', 'X')
load('C:\Users\Everybody\Desktop\Quentin_Pikeroen\Quentin_Pikeroen\Code\Positions\New_Conditions_2\1W700C15B\position.mat', 'Y')

for ii=1:1%length(X(:,1))
    figure(1)
    voronoi(X(ii,:),Y(ii,:))
    xlim([0 1266])
    ylim([0 1075])
    drawnow
end

% [vx, vy] = voronoi(X(1,:),Y(1,:));
% polyarea(vx,vy)

x = rand(10,1) ;  y = rand(10,1) ;
[v,c] = voronoin([x y]) ;
figure
hold on
voronoi(x,y)
A = zeros(length(c),1) ;
for i = 1:length(c)
    v1 = v(c{i},1) ; 
    v2 = v(c{i},2) ;
    patch(v1,v2,rand(1,3))
    A(i) = polyarea(v1,v2) ;
end


