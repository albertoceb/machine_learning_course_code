function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
h_theta = sigmoid(X*theta);


first_half = -1*(y.*log(h_theta));

second_half = (1-y).*log(1-h_theta);

cost_reg = (lambda*(sum(theta.^2) - theta(1)^2))/(2*m);

only_cost = sum(first_half - second_half )/m;

J = cost_reg + only_cost;
grad = zeros(size(theta));

theta_zero = theta(1);
thetas = X'*(h_theta - y) + lambda*theta;
grad = thetas/m;
grad(1) = grad(1) - (lambda*theta_zero)/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
