function  [Bx,By,Cx,Cy,Dx,Dy] = move3Robots(Ax,Ay,Bx,By,Cx,Cy,Dx,Dy)
% We take the centroid of the three points, and we move them so that the
% new centroid is on the point A

% Find the centroid of the 3 points
x1 = [Bx Cx Dx];
y1 = [By Cy Dy];
polyin = polyshape(x1,y1);
[x,y] = centroid(polyin);

%plot(polyin)
% figure;
% plot(Ax,Ay,'.','MarkerSize',20,'Color','r')
% hold on
% plot([Bx,Cx,Dx],[By,Cy,Dy],'.','MarkerSize',20,'Color','b')

Bx = Bx + Ax - x; 
By = By + Ay - y;
Cx = Cx + Ax - x; 
Cy = Cy + Ay - y;
Dx = Dx + Ax - x; 
Dy = Dy + Ay - y;
   
% plot([Bx,Cx,Dx],[By,Cy,Dy],'.','MarkerSize',20,'Color','g')
% hold off

end