function [predict_y,s,sub_y] = predict_lssvm_r(test_x,model)

[te_n,d] = size(test_x);


K_test = kernel_RBF(test_x,model.Xsv,model.gamma);

sub_y = K_test*(model.alpha)+ model.b;
	s = sigmod_f(sub_y);
predict_y = sign(sub_y);

end



%RBF kernel function
function k = kernel_RBF(X,Y,gamma)
	r2 = repmat( sum(X.^2,2), 1, size(Y,1) ) ...
	+ repmat( sum(Y.^2,2), 1, size(X,1) )' ...
	- 2*X*Y' ;
	k = exp(-r2*gamma); % RBFºË¾ØÕó
end


function s = sigmod_f(x)

s = 1./(1 + exp(-1*x));

end