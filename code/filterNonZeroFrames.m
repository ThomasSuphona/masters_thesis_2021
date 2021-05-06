function nonZeroFramesIndices = filterNonZeroFrames(iATrajs)

    nbrFrames = size(iATrajs, 2);
    
    zpos = find(~[0 iATrajs 0]);
    [~, grpidx] = max(diff(zpos));
    
    nonZeroFramesIndices = [zpos(grpidx):zpos(grpidx+1)-2];
end