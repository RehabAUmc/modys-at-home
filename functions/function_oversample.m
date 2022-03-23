function [X_train, y_train, D_train] = function_oversample(X_train, y_train, D_train, k)

%%% Oversample training data. This piece of code oversamples classes with
%%% a low number of samples, making sure all classes will have an equal amount of
%%% samples.

% Find indices for each class in the training data
idx_0_train = find(D_train{k}.class == '0');   
idx_1_train = find(D_train{k}.class == '1');
idx_2_train = find(D_train{k}.class == '2');
idx_3_train = find(D_train{k}.class == '3');
idx_4_train = find(D_train{k}.class == '4');

idx_all_train = {idx_0_train idx_1_train idx_2_train idx_3_train idx_4_train};
 
samples_per_class_training = [numel(idx_0_train) numel(idx_1_train) numel(idx_2_train) numel(idx_3_train) numel(idx_4_train)];
[max_samples,idx_max] = max([numel(idx_0_train) numel(idx_1_train) numel(idx_2_train) numel(idx_3_train) numel(idx_4_train)]);
idx_nonzero = find([numel(idx_0_train) numel(idx_1_train) numel(idx_2_train) numel(idx_3_train) numel(idx_4_train)] > 0);
idx_oversample = idx_nonzero(idx_nonzero~=idx_max);
 
clear D_train_extra
for i = idx_oversample
   D_train_extra{i,1} = [repmat(idx_all_train{i},floor((max_samples - samples_per_class_training(i)) / numel(idx_all_train{i})),1)];
   D_train_extra{i,1} = [D_train_extra{i,1} ; idx_all_train{i}(1:(max_samples - samples_per_class_training(i))-length(D_train_extra{i,1}))];
end

% Updates the input, output and dataframe with the oversamples data
if exist('D_train_extra') == 1
   idx_add = vertcat(D_train_extra{:});
   X_train{k} = [X_train{k} ; X_train{k}(idx_add,:)];
   y_train{k} = [y_train{k} ; y_train{k}(idx_add,:)];
   D_train{k} = [D_train{k} ; D_train{k}(idx_add,:)];
end

end
