function [F1_val_kfold, F1_test_kfold, accuracy_val_kfold, accuracy_test_kfold, recall_val_mean_kfold, precision_val_mean_kfold, recall_test_mean_kfold, precision_test_mean_kfold,  net_save, optimal_hiddenlayers_val]   = function_deep_learning(ext_idx, X, class, trainInd_DL, valInd_DL, testInd_DL)

% Define the input (x) and the output (t)
X = X';
t = dummyvar(class)';

% Train and evaluate

clear accuracy accuracy_test
% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.


%%% Train DL models with 1 to 40 hidden layers and find the model with the
%%% optimum number of hidden layers

for i = 1:40

    % Create a Pattern Recognition network
    hiddenLayerSize = i;
    net = patternnet(hiddenLayerSize, trainFcn);
    net.trainParam.showWindow = 0;

    % Setup Division of Data for Training, Validation, Testing
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInd_DL{1,ext_idx};
    net.divideParam.valInd   = valInd_DL{1,ext_idx};
    net.divideParam.testInd  = testInd_DL{1,ext_idx};
    net.performFcn = 'mse';

    % Train the network
    [net,tr] = train(net,X,t);

    % Test the network
    y = net(X);
    e = gsubtract(t,y);
    performance = perform(net,t,y);
    tind = vec2ind(t);  % targets
    yind = vec2ind(y);  % predictions
    percentErrors = sum(tind ~= yind)/numel(tind);
    accuracy(i) = round((1-percentErrors)*100,1);
    confMat_all = confusionmat(yind,tind);
%     recall_all  = confMat_all(1,1) / sum(confMat_all(1,:));
%     precision_all = confMat_all(1,1) / sum(confMat_all(:,1));
    [recall_all_mean, precision_all_mean] = function_calc_mean_recall_precision(confMat_all);
    F1_all(i) = 2*recall_all_mean*precision_all_mean / (recall_all_mean+precision_all_mean);
    
   
    % train accuracy
    tind_train = vec2ind(t(:,trainInd_DL{1,ext_idx}));  % targets
    yind_train = vec2ind(y(:,trainInd_DL{1,ext_idx}));  % prediction
    percentErrors_train = sum(tind_train ~= yind_train)/numel(tind_train);
    accuracy_train(i) = round((1-percentErrors_train)*100,1);
    confMat_train = confusionmat(yind_train,tind_train);
    [recall_train_mean, precision_train_mean] = function_calc_mean_recall_precision(confMat_train);
    F1_train(i) = 2*recall_train_mean*precision_train_mean / (recall_train_mean+precision_train_mean);
    
    % validation accuracy
    tind_val = vec2ind(t(:,valInd_DL{1,ext_idx}));  % targets
    yind_val = vec2ind(y(:,valInd_DL{1,ext_idx}));  % prediction
    percentErrors_val = sum(tind_val ~= yind_val)/numel(tind_val);
    accuracy_val(i) = round((1-percentErrors_val)*100,1);
    confMat_val = confusionmat(yind_val,tind_val);
    [recall_val_mean, precision_val_mean] = function_calc_mean_recall_precision(confMat_val);
    F1_val(i) = 2*recall_val_mean*precision_val_mean / (recall_val_mean+precision_val_mean);
    
    % test accuracy
    tind_test = vec2ind(t(:,testInd_DL{1,ext_idx}));  % targets
    yind_test = vec2ind(y(:,testInd_DL{1,ext_idx}));  % prediction
    percentErrors_test = sum(tind_test ~= yind_test)/numel(tind_test);
    accuracy_test(i) = round((1-percentErrors_test)*100,1);
    confMat_test = confusionmat(yind_test,tind_test);
    [recall_test_mean, precision_test_mean] = function_calc_mean_recall_precision(confMat_test)
    F1_test(i) = 2*recall_test_mean*precision_test_mean / (recall_test_mean+precision_test_mean)
   

    % Save the highest test accuracy achieved during the 40
    % iterations
    if i == 1 
        %accuracy_test_save = accuracy_test(1);
        F1_val_save = F1_val(1);
        net_save = net;
        
        tr_test = tr;
        y_save = y;
    end
    if F1_val(i) > max(F1_val(1:i-1))
        F1_val_save = F1_val(i);
        net_save = net;
        tr_test = tr;
        y_save = y;
    end

    % If a F1 validation score of 1 is achieved during one of the 40 iterations,
    % stop the rest of the iterations
    if exist('accuracy_test_save') == 1
        if F1_val_save == 1;
           break
        end
    end

end

% Find the optimal number of hidden layers (with the highest val f1 score)
optimal_hiddenlayers_val{1,ext_idx}   = find(F1_val==max(F1_val))

% highest accuracies  
highest_accuracy       = max(accuracy)
highest_accuracy_train = max(accuracy_train);
highest_accuracy_val   = max(accuracy_val)
highest_accuracy_test  = max(accuracy_test);
highest_F1_val         = max(F1_val);

% Find the total accuracy and validation accuracy belonging to the optimal
% amount of hidden layers
accuracy_all = accuracy(optimal_hiddenlayers_val{1,ext_idx}(1))
accuracy_val = accuracy_val(optimal_hiddenlayers_val{1,ext_idx}(1))
accuracy_test = accuracy_test(optimal_hiddenlayers_val{1,ext_idx}(1))



%%% Train the optimal model

for k=1:5;

    % Create a Pattern Recognition network
    hiddenLayerSize = optimal_hiddenlayers_val{1,ext_idx}(1);    % optimum number of hidden layers
    net = net_save;

%     % Test the network
%     y = net(x);
%     e = gsubtract(t,y);
%     performance = perform(net,t,y);
%     tind = vec2ind(t);
%     yind = vec2ind(y);
%     percentErrors = sum(tind ~= yind)/numel(tind);
%     accuracy = round((1-percentErrors)*100,1);

    % Validation accuracy
    tind_val                = vec2ind(t(:,valInd_DL{k,ext_idx}));  % targets
    yind_val                = vec2ind(y(:,valInd_DL{k,ext_idx}));  % prediction
    percentErrors_val       = sum(tind_val ~= yind_val)/numel(tind_val);
    accuracy_val_kfold(k)   = round((1-percentErrors_val)*100,1);
    confMat_val             = confusionmat(yind_val,tind_val);
    [recall_val_mean, precision_val_mean] = function_calc_mean_recall_precision(confMat_val);
    recall_val_mean_kfold(k)    = recall_val_mean;
    precision_val_mean_kfold(k) = precision_val_mean;    
    F1_val_kfold(k)         = 2*recall_val_mean*precision_val_mean / (recall_val_mean+precision_val_mean);
    
    % Test accuracy
    tind_test               = vec2ind(t(:,testInd_DL{k,ext_idx}));  % targets
    yind_test               = vec2ind(y(:,testInd_DL{k,ext_idx}));  % prediction
    percentErrors_test      = sum(tind_test ~= yind_test)/numel(tind_test);
    accuracy_test_kfold(k)  = round((1-percentErrors_test)*100,1);
    confMat_test            = confusionmat(yind_test,tind_test);
    [recall_test_mean, precision_test_mean] = function_calc_mean_recall_precision(confMat_test);
    recall_test_mean_kfold(k)    = recall_val_mean;
    precision_test_mean_kfold(k) = precision_val_mean;  
    F1_test_kfold(k)        = 2*recall_test_mean*precision_test_mean / (recall_test_mean+precision_test_mean);
    
end

end
