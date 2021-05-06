function msdr = msd(X, Y,nImages)
   
    L = length(X);
    
    msdx = zeros(L-1, 1);
    msdy = zeros(L-1, 1);
    msdr = zeros(L-1, 1);
  
    
    for n = 1:L-1
        msdx(n, 1) = sum((X(1+n:end)- X(1:end-n)).^2)/(L-n);
        msdy(n, 1) = sum((Y(1+n:end)- Y(1:end-n)).^2)/(L-n);
        
        msdr(n, 1) = msdx(n, 1) + msdy(n, 1);
    end
    for n = L:nImages-1
        msdr(n, 1) = 0;
    end
        
    
end