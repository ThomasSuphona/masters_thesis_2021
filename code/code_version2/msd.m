function msdr = msd(X, Y)
   
    nbrFrames = length(X);
    
    msdx = zeros(nbrFrames-1, 1);
    msdy = zeros(nbrFrames-1, 1);
    msdr = zeros(nbrFrames-1, 1);
  
    
    for n = 1:nbrFrames-1
        msdx(n, 1) = sum((X(1+n:end)- X(1:end-n)).^2)/(nbrFrames-n);
        msdy(n, 1) = sum((Y(1+n:end)- Y(1:end-n)).^2)/(nbrFrames-n);
        
        msdr(n, 1) = msdx(n, 1) + msdy(n, 1);
    end
    
end