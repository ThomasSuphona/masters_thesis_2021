function [msdr, divFr] = msdAvg(X, Y)
   
    L = length(X);
    divFr = zeros(L-1, 1);
    
    msdx = zeros(L-1, 1);
    msdy = zeros(L-1, 1);
    msdr = zeros(L-1, 1);
  
    
    for n = 1:L-1
        for i = 1:L-n
            if X(i+n)==0 || X(i)==0
                msdx(n, 1) = 0;
            else
                msdx(n, 1) = msdx(n, 1) + (X(i+n)- X(i)).^2;
                msdy(n, 1) = msdy(n, 1) + (Y(i+n)- Y(i)).^2;
                
                divFr(n, 1) = divFr(n, 1) + 1;
            end
        end
        
        msdx(n, 1) = msdx(n, 1)/(L-n);
        msdy(n, 1) = msdy(n, 1)/(L-n);
        msdr(n, 1) = msdx(n, 1) + msdy(n, 1);
    end
end