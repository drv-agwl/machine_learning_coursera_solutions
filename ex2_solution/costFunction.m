function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%disp(grad);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

temp=(theta')*(X');
temp2=temp';
h=sigmoid(temp2);
sum=0;
for i=1:m
    sum=sum+y(i,1)*log(h(i,1))+(1-y(i,1))*log(1-h(i,1));
end
J=(-1/m)*sum;
temp3=h-y;
n=size(theta);
for i=1:n
    sum=0;
    for j=1:m
        sum=sum+temp3(j,1)*X(j,i);
    end
    sumx=(1/m)*sum;
    grad(i,1)=sumx;
end
% =============================================================

end
