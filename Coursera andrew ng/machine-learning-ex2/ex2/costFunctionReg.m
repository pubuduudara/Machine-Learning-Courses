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


J1=0;
for i=1:m
  theta_transpose_x=X(i,:)*theta;
  h_theta=sigmoid(theta_transpose_x);
  J1+= -1*y(i,1)*log(h_theta) -( (1-y(i,1))*log(1-h_theta));
end

J1=J1/m;

t=size(theta);
J2=0;
for i=2:t(1)
  J2+= theta(i,1)^2;
end

J2=(J2*lambda)/(2*m);
J=J1+J2;


k=size(theta);

for j=1:k
  t=0;
  for i=1:m
    theta_transpose_x=X(i,:)*theta;
    h_theta=sigmoid(theta_transpose_x);
    t+=(h_theta-y(i,1))*X(i,j);
    %grad(j,1)+= (h_theta-y(i,1))*X(i,j);
  end
  if j==1
    grad(j,1)=t/m;
  else
    grad(j,1)=t/m + (theta(j,1)*lambda)/m;
    endif
  end

% =============================================================

end
