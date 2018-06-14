function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
%disp(mu);
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

n=length(X(1,:));
%disp(n);
m=length(X(:,1));
%disp(m);
for i=1:n
    sum=0;
    %disp(X(:,i));
    %disp(X(:,i));
    maxi=max(X(:,i));
    mini=min(X(:,i));
    for j=1:m
        sum=sum+X(j,i);
    end
    mu(1,i)=sum/m;
    sigma(1,i)=std(X(:,i));
end
for i=1:n
    muu=mu(1,i);
    s=sigma(1,i);
    for j=1:m
        X_norm(j,i)=(X(j,i)-muu)/s;
    end
end
disp(X);
disp(X_norm);


% ============================================================

end
