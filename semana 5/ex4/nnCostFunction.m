function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = size(Theta2,1);
total = 0.0;
for i = 1:m
   a1 = [1; X(i,:)'];
   % a' para mejorar visibilidad en el ide
   z2 = Theta1 * a1;
   a2 = [1; sigmoid(z2)];
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);
   hx = a3;
   yi = zeros(K, 1);
   yi(y(i),1) = 1;

   first_half = -1*(yi.*log(hx));

   second_half = (1-yi).*log(1-hx);
   total = total + sum(first_half - second_half);
   
endfor
J = total/m;
cost_reg = 0.0;
cost_reg = sum((Theta1(:,[2:size(Theta1,2)]).^2)(:)) + sum((Theta2(:,[2:size(Theta2,2)]).^2)(:));

         
% You need to return the following variables correctly 

  J = total/m + (cost_reg/(2*m))*lambda;

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
   a1 = [1; X(t,:)'];
   % a' para mejorar visibilidad en el ide
   z2 = Theta1 * a1;
   a2 = [1; sigmoid(z2)];
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);
   hx = a3;

   yi = zeros(K, 1);
   yi(y(i),1) = 1;
   delta3 = a3 - yi
   zzzz = sigmoidGradient(z2);
   delta2 = Theta2(:,2:end)' * delta3 .* zzzz; % a' para mejorar visibilidad en el ide

   Delta1 = Delta1 + delta2*(a1');   % a' para mejorar visibilidad en el ide
   Delta2 = Delta2 + delta3*(a2');   % a' para mejorar visibilidad en el ide
   
   
endfor


Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

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



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
