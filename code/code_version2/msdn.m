function msdr = msdn(X, Y)
   
    nbrFrames = size(X, 1);
    nbrParticles = size(X, 2);
    
    msdx = zeros(nbrFrames-1, nbrParticles);
    msdy = zeros(nbrFrames-1, nbrParticles);
    msdr = zeros(nbrFrames-1, nbrParticles);
  
    
    for n = 1:nbrFrames-1
        msdx(n, :) = sum((X(1+n:end, :)- X(1:end-n, :)).^2)/(nbrFrames-n);
        msdy(n, :) = sum((Y(1+n:end, :)- Y(1:end-n, :)).^2)/(nbrFrames-n);
        
        msdr(n, :) = msdx(n, :) + msdy(n, :);
    end
    
    %msdrEnsemble = mean(msdr, 2);
end