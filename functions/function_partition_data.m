function [trainInd, testInd, trainInd_DL, testInd_DL, valInd_DL, size_y_test_DL, size_y_val_DL] = function_partition_data(dataframes)

% Train & test split
% trainRatio = 0.8;    % Use 80% of the samples for training
% testRatio  = 0.2;    % Use 20% of the samples for testing

for ext_idx = 1:6
    
clearvars -except ext_idx dataframes trainInd testInd 

names = {'max_Acc_X' ; 'max_Acc_X' ; 'max_Acc_X' ; 'max_Acc_X' ; 'max_AL_Acc_X' ; 'max_AL_Acc_X'};

D = dataframes{ext_idx,1};                               % Picks the dataframe belonging to the extremity index the code is looping over
nan = isnan(D{:,end});                                   % Finds the NaN values in the clinical scores
D = D(~nan,:);                                           % Remove the rows with NaN values as a clinical score
D.Properties.VariableNames{size(D,2)} = 'class';
D = movevars(D, 'class', 'Before', names{ext_idx,1});    % Puts the clinical score ('class') as the first column
D.class = categorical(D.class);                          % Transforms the clinical score to a categorical array

% Indices per class
idx_0 = find(D.class == categorical(0));    % Finds indices of class '0'
idx_1 = find(D.class == categorical(1));    % Finds indices of class '1'    
idx_2 = find(D.class == categorical(2));    % Finds indices of class '2'
idx_3 = find(D.class == categorical(3));    % Finds indices of class '3'
idx_4 = find(D.class == categorical(4));    % Finds indices of class '4'

%     % Delete class that has less than 5 samples
%     samples_per_class = [numel(idx_0) numel(idx_1) numel(idx_2) numel(idx_3) numel(idx_4)];
%     minimal_samples_per_class = 5;
%     idx_del = find(samples_per_class < minimal_samples_per_class);
%     for i = idx_del
%         classes = {'0' '1' '2' '3' '4'};
%         idx_del_class = find(D.class == categorical(classes(i)));
%         D(idx_del_class,:) = [];
%     end
%     
%     % Update the indices for each class
%     idx_0 = find(D.class == categorical(0));
%     idx_1 = find(D.class == categorical(1));
%     idx_2 = find(D.class == categorical(2));
%     idx_3 = find(D.class == categorical(3));
%     idx_4 = find(D.class == categorical(4));

% Randomly split the data for each class into training data (80%) and
% testing data (20%)
if numel(idx_0) > 5
    partitioning0 = cvpartition(numel(idx_0),'kFold',5);
end
if numel(idx_1) > 5 
    partitioning1 = cvpartition(numel(idx_1),'kFold',5);
end
if numel(idx_2) > 5
    partitioning2 = cvpartition(numel(idx_2),'kFold',5);
end
if numel(idx_3) > 5
    partitioning3 = cvpartition(numel(idx_3),'kFold',5);
end
if numel(idx_4) > 5
    partitioning4 = cvpartition(numel(idx_4),'kFold',5);
end


for k = 1:5     % 5-fold cross validation
    
clear train0 test0 train1 test1 train2 test2 train3 test3 train4 test4


if exist('partitioning0') == 1
    train0 = training(partitioning0,k);
    test0  = test(partitioning0,k);
else train0 = [];
    test0 = 1:length(idx_0);
end
if exist('partitioning1') == 1
    train1 = training(partitioning1,k);
    test1  = test(partitioning1,k);
else train1 = [];
    test1 = 1:length(idx_1);
end
if exist('partitioning2') == 1
    train2 = training(partitioning2,k);
    test2  = test(partitioning2,k);
else train2 = [];
    test2 = 1:length(idx_2);
end
if exist('partitioning3') == 1
    train3 = training(partitioning3,k);
    test3  = test(partitioning3,k);
else train3 = [];
    test3 = 1:length(idx_3);
end
if exist('partitioning4') == 1
    train4 = training(partitioning4,k);
    test4  = test(partitioning4,k);
else train4 = [];
    test4 = 1:length(idx_4);
end

% Create array with all training indices and arrat with all testing indices
trainInd_temp = [idx_0(train0); idx_1(train1); idx_2(train2); idx_3(train3); idx_4(train4)];
trainInd{k, ext_idx}   = trainInd_temp(randperm(numel(trainInd_temp)));
testInd_temp  = [idx_0(test0) ; idx_1(test1) ; idx_2(test2) ; idx_3(test3) ; idx_4(test4) ];
testInd{k, ext_idx}    = testInd_temp(randperm(numel(testInd_temp)));

end

end

for i = 1:5
    for j = 1:6
        trainInd_DL{i,j}  =  trainInd{i,j}(1:round(0.8*length(trainInd{i,j})));
        valInd_DL{i,j}    =  trainInd{i,j}(round(0.8*length(trainInd{i,j}))+1:end);
        testInd_DL{i,j}   =  testInd{i,j}(randperm(numel(testInd{i,j})));
        
        size_y_test_DL(i,j)  =  length(testInd_DL{i,j});
        size_y_val_DL(i,j)   =  length(valInd_DL{i,j});
    end
end

end
