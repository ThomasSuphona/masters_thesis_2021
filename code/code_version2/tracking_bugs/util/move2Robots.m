function [Bx,By,Cx,Cy] = move2Robots(Ax,Ay,Bx,By,Cx,Cy)
% This is equivalent to take the centroid of the points B and C, and to
% move those points for the centroid to move on the point A

    newBx = Ax + (Bx-Cx)/2;
    newBy = Ay + (By-Cy)/2;
    
    newCx = Ax - (Bx-Cx)/2;
    newCy = Ay - (By-Cy)/2;

    Bx = newBx;
    By = newBy;
    Cx = newCx;
    Cy = newCy;
    
%Tests    
%     figure;
%     plot(Ax,Ay,'.','MarkerSize',20,'Color','r')
%     hold on
%     plot([Bx,Cx],[By,Cy],'.','MarkerSize',20,'Color','b')
%     plot([newBx,newCx],[newBy,newCy],'.','MarkerSize',20,'Color','g')
%     hold off
end

