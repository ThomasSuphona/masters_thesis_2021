function nonZeroIndices = filterNonzero(aTrajs, na_true, na_tracked, tolFrames)
    
    nbrFrames = size(aTrajs, 2);
    A = zeros(na_tracked, 2);
    na_curr = na_tracked;
    
    
    A(:, 1) = sum(aTrajs==0, 2);
    A(:, 2) = nbrFrames;
    indx = find((A(:,1)./A(:,2))<tolFrames);
    na_curr = size(indx, 1);
    
    nonZeroIndices = indx;
    %A(indx, :)
end