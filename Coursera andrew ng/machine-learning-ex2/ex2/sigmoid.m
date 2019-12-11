function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[r,c]=size(z);
for row=1:r
  for col=1:c
    x=z(row,col);
    g(row,col)=1/(1+e^(-x));
  end
end





% =============================================================

end
