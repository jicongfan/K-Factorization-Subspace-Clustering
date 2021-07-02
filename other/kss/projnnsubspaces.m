% projnnsubspaces provides the projection for the nearest subspaces
% 
% [Y IDX inpnorm] = projnnsubspaces(SS,X,IDX)
%
%
%Output parameter:
% Y: the projected data, where size(Y,1) is the number of the data, size(Y,2) is the dimension of the data
% IDX: the cluster indexes of the nearest subspaces
% inpnorm: the maximum of the inner product norm
%
%Input parameters:
% SS: subspaces, where size(SS,1) is the dimension of the subspaces, size(SS,2) is the dimension of the data and size(SS,3) is the number of the clusters
% X: data, where the number of data is size(X,1) and the dimension of the data is size(X,2)
% IDX (optional): if you know the nearest subspaces indexes, please provide them.
%
%Version: 20120629

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ksubspaces                                               %
%                                                          %
% Copyright (C) 2012 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y IDX inpnorm] = projnnsubspaces(SS,X,IDX)

if( nargin < 3 )
 inpnorm = inpnormsubspaces(SS,X);
 [inpnormt IDX] = max(inpnorm,[],2);
elseif( nargout >= 3 )
 inpnorm = [];
end

Y=zeros(size(X));
for j=1:size(SS,3)
 Y(IDX==j,:) = X(IDX==j,:) * SS(:,:,j)' * SS(:,:,j);
end
