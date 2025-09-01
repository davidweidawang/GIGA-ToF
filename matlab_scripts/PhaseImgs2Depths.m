
% This function computes the final depth, given individual phases and the
% vector of frequencies. Also, given as input is the depth range - which is
% the set of possible depth values. Essentially it's doing phase unwrapping
% from multi-freq measurements. Somehow using 6 freqs from 10:30:160 was
% insufficient and led to a few outliers, since the method processes each 
% pixel individually. Finer freqs (more phaseImgs) helps disambiguate. 

function [Depths]   = PhaseImgs2Depths(freqVec, PhaseMaps, DepthRange, phaseOffsets)

PhaseMaps = permute(PhaseMaps,[2,3,1]);
[nr,nc,nFreq]   = size(PhaseMaps);

% Computing the phases for every candidate depth
CandidatePhases     = zeros(1, numel(DepthRange), nFreq);
for i=1:nFreq
    DepthRange_ = DepthRange;
    if exist('phaseOffsets', 'var')
        DepthRange_ = DepthRange + phaseOffsets(i);
    end
    CandidatePhases(1,:,i)      = mod(2* pi * 2 * DepthRange_ / (3e8/freqVec(i)), 2*pi);       % Multiply depth range by 2 because light traverses the distance twice
end
CandidatePhases     = repmat(CandidatePhases, [nr 1 1]);

% Computing the depths
Depths  = zeros(nr, nc);

% to find a depth that fits all phase images across a number of freqs
for i=1:nc                  % Consider one column at a time
%    i
    
    PhaseMapsTmp    = PhaseMaps(:,i,:);
    PhaseMapsTmp    = repmat(PhaseMapsTmp, [1 numel(DepthRange) 1]);
    
    ErrMat          = sum((PhaseMapsTmp - CandidatePhases).^2, 3);

%     keyboard
    [~, indices]   	= min(ErrMat, [], 2);
    
    Depths(:,i)     = DepthRange(indices);
end
