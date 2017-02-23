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
normalisation_term = lambda/2 * 1/m * (theta.^2);
normalisation_term(1) = 0;
J = 1/2 * 1/m * sum(sum((X*theta -y).^2)) + sum(normalisation_term);
normalisation_term_grad = lambda/m * theta;
normalisation_term_grad(1) = 0;
grad = (1/m * X'*(X*theta - y)) + normalisation_term_grad;
% =========================================================================

grad = grad(:);

end
