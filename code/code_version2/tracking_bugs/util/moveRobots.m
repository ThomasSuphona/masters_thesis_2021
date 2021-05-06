function  varargout = moveRobots(Ax,Ay,varargin)
% We take the centroid of all the points, and we move them so that the
% new centroid is on the point A

% Check if the input and output is coherent
if nargout ~= nargin - 2
    error(['The number of points to move is not the same in '...
        'output than in input'])
end

% Find the centroid of all the points
x1 = [];
y1 = [];
nargin
for ii = 1:(nargin-2)/2
    x1 = [x1 varargin{2*ii -1}];
    y1 = [y1 varargin{2*ii}];
end
x = mean(x1);
y = mean(y1);

%Move the points so their new centroid is now (Ax,Ay)
for ii = 1:nargout
    %j=1 for x coordinate and j=2 for y coordinate
    j = 2-mod(ii,2); 
    if j==1
        varargout{ii} = varargin{ii} + Ax - x;
    elseif j==2
        varargout{ii} = varargin{ii} + Ay - y;
    end
end


% %Plot
% figure;
% plot(Ax,Ay,'.','MarkerSize',20,'Color','r')
% hold on
% plot(x,y,'*','MarkerSize',20,'Color','b')
% plot(x1,y1,'.','MarkerSize',20,'Color','b')
% 
% Px=[];
% Py=[];
% for ii=1:nargout
%     if mod(ii,2) == 1 
%         Px = [Px varargout{ii}];
%     elseif mod(ii,2) == 0
%         Py = [Py varargout{ii}];
%     end
% end
% plot(Px,Py,'.','MarkerSize',20,'Color','g')
% hold off

end