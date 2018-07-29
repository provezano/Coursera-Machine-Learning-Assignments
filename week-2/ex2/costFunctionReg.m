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

for i = 1:m
	J = J - y(i,1)*log(sigmoid(dot(theta, X(i,:)))) - (1 - y(i,1))*log(1 - sigmoid(dot(theta, X(i,:))));
	for j = 2:length(grad)
		J = J + (lambda/(2*m)*(theta(j,1)^2));
	endfor
endfor

J = J/m;

for j = 1:size(grad)(1)
	for i = 1:m
		grad(j,1) = grad(j,1) + (sigmoid(dot(theta, X(i,:))) - y(i, 1))*X(i,j);

		if (j > 1)
			grad(j,1) = grad(j,1) + lambda/m*theta(j,1);
		endif
	endfor
	grad(j,1) = grad(j,1)/m;
endfor

% =============================================================

end
