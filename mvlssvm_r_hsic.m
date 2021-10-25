function [predict_y,Scores,models,real_y] = mvlssvm_r_hsic(train_x,feature_id,train_y,test_x,c_list,gamma_list,intermax)

%fprintf('Multi-View LS-SVM via HSIC \n');
predict_y=[];
Scores=[];



V = size(feature_id,1);
%V=3;
num_train_samples = size(train_x,1);
num_test_samples = size(test_x,1);
models=[];
E = [];
G_list=[];
G_complement= zeros(num_train_samples,num_train_samples);
l=ones(num_train_samples,1);
H = eye(num_train_samples) - (l*l')/num_train_samples;

lamda_list=[1,1,1,0.1,1,0.1];

%Initialize the model of each view
for i=1:V
	train_v_x = train_x(:,feature_id(i,1):feature_id(i,2));
   % train_v_x = train_x;
	parameters_model.gamma = gamma_list(i);
	parameters_model.c = c_list(i);
	[model_v] = train_lssvm_r(train_v_x,train_y,parameters_model);
	models{i}=model_v;
	e_v = (1/model_v.c)*model_v.alpha;
	E(:,i) = e_v;
    
	G_complement = G_complement*0;
	for k=1:V
            if (k==i) 
                continue;
            end
            G_complement =  G_complement + H*e_v*e_v'*H;                    
    end
	G_list(:,:,i) = lamda_list(i)* G_complement;
end




%Updata G
for o=1:intermax
	for i=1:V
		train_v_x = train_x(:,feature_id(i,1):feature_id(i,2));
        %train_v_x = train_x;
		parameters_model.gamma = gamma_list(i);
		parameters_model.c = c_list(i);
		[model_v] = train_lssvm_r_g(train_v_x,train_y,parameters_model,G_list(:,:,i));
		models{i}=model_v;
		C = eye(num_train_samples)*parameters_model.c + G_list(:,:,i);
		C = inv(C);
		e_v = C*model_v.alpha;
		E(:,i) = e_v;
		G_complement = G_complement*0;
		for k=1:V
				if (k==i) 
					continue;
				end
				G_complement =  G_complement + H*e_v*e_v'*H;                    
		end
		G_list(:,:,i) = lamda_list(i)*G_complement;
	end
end


beta = 1/V;
yy = 0;
real_y=[];                             
	for i=1:V
        %[predict_y,s,sub_y] = predict_lssvm_r(test_x,models{i});
		[predict_y,s,sub_y] = predict_lssvm_r(test_x(:,feature_id(i,1):feature_id(i,2)),models{i});
		yy = yy + sub_y*beta;
		real_y=[real_y,sub_y];
	end
	Scores = sigmod_f(yy);
	predict_y = sign(yy);
	

end



function s = sigmod_f(x)

s = 1./(1 + exp(-1*x));

end



