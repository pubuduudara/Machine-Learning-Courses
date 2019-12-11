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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for i=1:m
  theta_transpose_x=X(i,:)*theta;
  h_theta=sigmoid(theta_transpose_x);
  J+= -1*y(i,1)*log(h_theta) -( (1-y(i,1))*log(1-h_theta));
end

J=J/m;

k=size(theta);

for j=1:k
  t=0;
  for i=1:m
    theta_transpose_x=X(i,:)*theta;
    h_theta=sigmoid(theta_transpose_x);
    t+=(h_theta-y(i,1))*X(i,j);
    %grad(j,1)+= (h_theta-y(i,1))*X(i,j);
  end
  grad(j,1)=t/m;
  end





% =============================================================

end
