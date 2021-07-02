function X=normalizeL2(X)
X=X./repmat(sum(X.^2).^0.5,size(X,1),1);
end