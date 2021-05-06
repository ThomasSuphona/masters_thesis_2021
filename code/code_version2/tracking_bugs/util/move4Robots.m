function  [Bx,By,Cx,Cy,Dx,Dy,Ex,Ey] = move4Robots(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Ex,Ey)
% We take the centroid of the 4 points, and we move them so that the
% new centroid is on the point A

% Find the centroid of the 3 points
x1 = [Bx Cx Dx Ex];
y1 = [By Cy Dy Ey];
polyin = polyshape(x1,y1);
[x,y] = centroid(polyin);

% %plot(polyin)
% figure;
% plot(Ax,Ay,'.','MarkerSize',20,'Color','r')
% hold on
% plot([Bx,Cx,Dx,Ex],[By,Cy,Dy,Ey],'.','MarkerSize',20,'Color','b')

Bx = Bx + Ax - x; 
By = By + Ay - y;
Cx = Cx + Ax - x; 
Cy = Cy + Ay - y;
Dx = Dx + Ax - x; 
Dy = Dy + Ay - y;
Ex = Ex + Ax - x; 
Ey = Ey + Ay - y;
   
% plot([Bx,Cx,Dx,Ex],[By,Cy,Dy,Ey],'.','MarkerSize',20,'Color','g')
% hold off

end