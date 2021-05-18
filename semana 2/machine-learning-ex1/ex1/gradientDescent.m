function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
number_of_theta = length(theta);
temp_theta = [];
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

A = [X y];
b = [theta; -1];

C = A * b;
theta0 = theta(1,1) - (alpha/m)*sum(C.*X(:,1));
theta1 = theta(2,1) - (alpha/m)*sum(C.*X(:,2));

theta(1,1) = theta0;
theta(2,1) = theta1;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
