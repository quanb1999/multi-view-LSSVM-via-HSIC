function [predict_y,Scores,models,real_y] = EnsembleLssvm_r(train_x,feature_id,train_y,test_x,c_list,gamma_list)

%fprintf('Ensemble LS-SVMs \n');
predict_y=[];
Scores=[];



V = size(feature_id,1);
%V=4;
num_train_samples = size(train_x,1);
num_test_samples = size(test_x,1);
models=[];
E = [];
G_list=[];
G_complement= zeros(num_train_samples,num_train_samples);;
l=ones(num_train_samples,1);
H = eye(num_train_samples) - (l*l')/num_train_samples;

%Initialize the model of each view
for i=1:V
	train_v_x = train_x(:,feature_id(i,1):feature_id(i,2));
	parameters_model.gamma = gamma_list(i);
	parameters_model.c = c_list(i);
	[model_v] = train_lssvm_r(train_v_x,train_y,parameters_model);
	models{i}=model_v;
	
end

beta = 1/V;
yy = 0;
real_y=[];
	for i=1:V
		[predict_y,s,sub_y] = predict_lssvm_r(test_x(:,feature_id(i,1):feature_id(i,2)),models{i});
		yy = yy + sigmod_f(sub_y)*beta;
		
        real_y=[real_y,sub_y];
	end
	%Scores = sigmod_f(real_y);
    Scores = yy;
	predict_y = ones(num_test_samples,1);
	predict_y(find(yy<0.5))=-1;
	

end



function s = sigmod_f(x)

s = 1./(1 + exp(-1*x));

end



