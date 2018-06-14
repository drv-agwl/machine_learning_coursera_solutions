function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
n=length(theta);
%disp(theta);
%disp(X);
m=length(X*theta);
%disp(X*theta);
temp=log(sigmoid(X*theta));
ans1=((y')*temp);
temp2=ones(length(y),1)-y;
temp4=log(ones(length(X*theta),1)-sigmoid((X*theta)));
ans2=((temp2')*temp4);
J=(1/m)*((-1)*ans1-ans2);
sum=((theta')*theta);
sum=sum-theta(1,1)*theta(1,1);
J=J+(lambda/(2*m))*sum;
% =============================================================

temp5=sigmoid(X*theta)-y;
grad=(1/m)*((X')*temp5) + (lambda/m)*theta;
grad(1,1)=grad(1,1)-(lambda/m)*theta(1,1);
grad = grad(:);

end
