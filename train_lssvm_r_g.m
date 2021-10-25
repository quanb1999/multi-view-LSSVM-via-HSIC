function [model] = train_lssvm_r_g(train_x,train_y,parameters_model,G)

[tr_n,d] = size(train_x);


l_vector = ones(tr_n,1);

Garm_M = zeros(tr_n,tr_n);
Garm_M = kernel_RBF(train_x,train_x,parameters_model.gamma);


C = eye(tr_n)*parameters_model.c +  G;
C = inv(C);

P = [0,l_vector';l_vector,Garm_M + C];
Q = [0;train_y];

alpha = P\Q;

	b = alpha(1,1);
	alpha(1) = [];
	model.alpha = alpha;
	model.b = b;
	model.Ysv = train_y;
	model.Xsv = train_x;
	model.gamma = parameters_model.gamma;
	model.c = parameters_model.c;
end



%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); % RBFºË¾ØÕó
end