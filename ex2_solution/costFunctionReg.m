function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

temp=(theta')*(X');
temp2=temp';
h=sigmoid(temp2);
sum=0;
for i=1:m
    sum=sum+y(i,1)*log(h(i,1))+(1-y(i,1))*log(1-h(i,1));
end
n=size(theta);
sumx=0;
for i=2:n
    sumx=sumx+theta(i,1)*theta(i,1);
end
J=(-1/m)*sum + (lambda/(2*m))*sumx;
temp3=h-y;
sum2=0;
for i=1:m
    sum2=sum2+temp3(i,1)*X(i,1);
end
grad(1,1)=(sum2/m);
for i=2:n
    sum=0;
    for j=1:m
        sum=sum+temp3(j,1)*X(j,i);
    end
    sumx=(1/m)*sum;
    grad(i,1)=sumx + (lambda/m)*theta(i,1);
end




% =============================================================

end
