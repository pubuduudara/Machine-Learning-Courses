function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%disp(size(X)); % 12X2
%disp(size(y)); % 12X1
%size(theta)= 2X1

%h_theta=X*theta;
%j1= (sum((h_theta-y).^2))/(2*m);
%j2=sum(theta(2:end,:).^2)*lambda/(2*m);
%J=j1+j2;
%
%
%cost = (X*theta-y);
%theta2 = theta;
%grad = 1/m*((X'*cost)+theta2*lambda);

cost = (X*theta-y);
theta2 = theta;
theta2(1) = 0; 

J = 1/(2*m)*(sum(cost.^2)+lambda*(sum(theta2.^2)));

grad = 1/m*((X'*cost)+theta2*lambda);

% =========================================================================

grad = grad(:);

end
