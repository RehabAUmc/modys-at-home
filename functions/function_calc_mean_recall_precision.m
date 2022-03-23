function [recall_mean, precision_mean] = function_calc_mean_recall_precision(confMat)

for i = 1:size(confMat,1)
    recall(i)    = confMat(i,i) / sum(confMat(i,:));
    precision(i) = confMat(i,i) / sum(confMat(:,i));
end

recall(isnan(recall)) = 0;
precision(isnan(precision)) = 0;

recall_mean = mean(recall);
precision_mean = mean(precision);

end
