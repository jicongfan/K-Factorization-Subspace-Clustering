% ksubspaces provides k-subspaces clustering with k-means like algorithm
% 
% [IDX, SS, inpnorm] = ksubspaces(X,k,dim,SS,iter,ep)
%
%
%Output parameter:
% IDX: cluster indexes of each data
% SS:  basis vectors which represent subspacess
% inpnorm: maximum norm in projected subspacess
%
%Input parameters:
% X: data, where the number of data is size(X,1) and the dimension of the data is size(X,2)
% k: the number of the clusters
% dim: the dimension of the subspacess
% SS: initial subspaces, where size(SS,1) is the dimension of the subspaces, size(SS,2) is the dimension of the data and size(SS,3) is the number of the clusters
% iter (optional): maximum number of the iteration (default: 128)
% ep (optional): stoping criteria of the iteration (default: 128)
%
%
%Example:
% X = randn(1024,128);
% [IDX SS] = ksubspaces(X,16,2);
% Y = projnnsubspaces(SS,X);
%
%Version: 20120629

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ksubspaces                                               %
%                                                          %
% Copyright (C) 2012 Masayuki Tanaka. All rights reserved. %
%                    mtanaka@ctrl.titech.ac.jp             %
%                                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [IDX, SS, inpnorm] = ksubspaces(X,k,dim,SS,iter,ep)

if( nargin < 4 )
 SS = [];
end

if( nargin < 5 )
 iter = 128;
end

if( nargin < 6 )
 ep = 1E-3;
end

if( isempty(SS) )
 SS = randn(dim,size(X,2),k);
 for i=1:k
  SS(:,:,i) = orth(SS(:,:,i)')';
 end
end

SS0 = SS;

for i=1:iter
 inpnorm = inpnormsubspaces(SS,X);
 [inpnorm IDX] = max(inpnorm,[],2);
 for j=1:k
  p = (IDX==j);
  if( sum(p(:)) > dim )
   XX = X(p,:);
   [V,D] = eig(XX'*XX);
   V = V';
   V = flipud(V);
   SS(:,:,j) = V(1:dim,:);
  else
   SS(:,:,j) = orth(randn(dim,size(X,2))')';
  end
  for m=1:size(SS,1)
   if( sum(SS(m,:,j)) < 0 )
    SS(m,:,j) = -SS(m,:,j);
   end
  end
 end
 
 d = abs(SS-SS0);
 d = max(d(:));
% fprintf('%3d: %g\n', i, d);
 if( d < ep )
  break;
 end
 SS0 = SS;
end

