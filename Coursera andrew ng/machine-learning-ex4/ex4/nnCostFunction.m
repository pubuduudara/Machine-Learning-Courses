function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 


J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end

%disp(size(y)); =5000 1
%disp(size(X));=5000    401
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

a1=X;
z2=Theta1*X'; % size = 25 5000
a2=sigmoid(z2); % size = 25 5000
a2=[ones(1,m);a2]; % size = 26 5000
z3=Theta2 * a2; %size = 10 5000
a3=sigmoid(z3);

h_theta=a3'; % size 5000 10

j1=0;
for i=1:m
  ans1=0;
  for k=1:num_labels
    ans1+= (-1*yVec(i,k)) .* log(h_theta(i,k))-((1-yVec(i,k)).*log(1-h_theta(i,k)));
    %(-1*y).*log(h_theta)-((1-y).*log(1-h_theta));
  end
  j1+=ans1;
end

j1=j1/m;




j2=0;
%calculation for regularized version

r1=size(Theta1)(1);
c1=size(Theta1)(2)-1;
r2=size(Theta2)(1);
c2=size(Theta2)(2)-1;

j2_1=0;
for r=1:r1
  ans1=0;
  for c=1:c1
    ans1+=Theta1(r,c)^2;
  end
  j2_1+=ans1;
end

j2_2=0;
for r=1:r2
  ans1=0;
  for c=1:c2
    ans1+=Theta2(r,c)^2;
  end
  j2_2+=ans1;
end


j2=(lambda*(j2_1+j2_2))/(2*m);


regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J=j1+regularator;


  










% -------------------------------------------------------------

% =========================================================================



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%disp(size(yVec))
%disp(size(X));=5000    401
%size Yvec =5000 10
%backpropogation algorithm
capital_delta_2=0;
capital_delta_1=0;
for t=1:m
  a1=X(t,:); % 1x401
  z2=Theta1*a1'; % size = 25 1
  a2=sigmoid(z2); % size = 25 1
  %a2=[ones(1,m);a2]; % size = 26 1
  a2=[1;a2];
  z3=Theta2 * a2; %size = 10 1
  a3=sigmoid(z3); %size = 10 1
  %disp(size(a3));
  delta_3=a3- yVec(t,:)';
  %disp(size(Theta2));%10   26
  %disp(size(delta_3)); %10   1
  z2=[1;z2];
  delta_2=(Theta2' * delta_3) .*sigmoidGradient(z2);
  %disp(size(delta_2));
  delta_2=delta_2(2:end,:);
  %disp(size(delta_2)); % 25x1
  capital_delta_1+=delta_2* a1;
  capital_delta_2+=delta_3* a2';
  
  end

capital_delta_1=capital_delta_1/m;
capital_delta_2=capital_delta_2/m;
%disp(size(Theta1));
%disp(size(Theta2));
%disp(size(capital_delta_2)); 10   26
%disp(size(capital_delta_1)); %25 401 

% ####################################
% apply regularization
Theta1_grad=(lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad= (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

capital_delta_1+=Theta1_grad;
capital_delta_2+=Theta2_grad;


grad=[capital_delta_1(:);capital_delta_2(:)];
end
