clear
load('Data_sets_PDB1075_186.mat');


train_X = [GE_1075,MCD_1075,NMBAC_1075,PSSM_AB_1075,PSSM_Pse_1075,PSSM_DWT_1075];
test_X = [GE_186,MCD_186,NMBAC_186,PSSM_AB_186,PSSM_Pse_186,PSSM_DWT_186];
COM_X = [train_X;test_X];
COM_X = line_map(COM_X);
train_X_S = COM_X(1:1075,:);
test_X_S = COM_X(1076:end,:);


gamma_list = [2^-1,2^-5,2^-1,2^-0,2^-0,2^-4];
c_list = [2^2,2^2,2^-0,2^-0,2^-0,2^-0];
% gamma_list = [2^3,2^0,2^0,2^0,2^0,2^0];
% c_list = [2^-5,2^0,2^0,2^0,2^0,2^0];

lammda =0.001;
mu = 0.000;gamma=2^-3;g_nn=10;pro=1.1;
feature_id=[1,150;151,1032;1033,1232;1233,1432;1433,1652;1653,2692];
intermax=3;

[predict_y,Scores,models,real_y] = mvlssvm_r_hsic(train_X_S,feature_id,label_1075,test_X_S,c_list,gamma_list,intermax);
%[predict_y,s,models,real_y] = EnsembleLssvm_r(train_X_S,feature_id,label_1075,test_X_S,c_list,gamma_list);
[ACC,SN,Spec,PE,NPV,F_score,MCC] = roc( predict_y,label_186 )

%[predict_y,Scores,models,real_y] = mvlssvm_r_hsic(train_X_S,feature_id,label_1075,train_X_S,c_list,gamma_list,intermax);
%[ACC,SN,Spec,PE,NPV,F_score,MCC] = roc( predict_y,label_1075 )




