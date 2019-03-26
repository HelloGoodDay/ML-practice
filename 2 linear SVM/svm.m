clear;

% ------------ UCI Dataset Attribute Information -------------------------
%  name: iris
%   1. sepal length in cm
%   2. sepal width in cm
%   3. petal length in cm
%   4. petal width in cm
%   5. class: 
%      -- Iris Setosa      -> class 1
%      -- Iris Versicolour -> class 0
%      -- Iris Virginica   -> class 0
% ------------------------------------------------------------------------

% read UCI data
data0 = textread('../data/iris.data','%s','delimiter',',');
data0 = reshape(data0, 5, 150);
[col row] = size(data0);
datas= cellfun(@(x) sscanf(x, '%f'), data0(1:4, :));
datas(col,:) = -1;
for i = 1:row
    switch cell2mat(data0(5, i))
        case('Iris-setosa')
            a = 1;
        case('Iris-versicolor')
            a = 0;
        case('Iris-virginica')
            a = 0;
    end
    datas(col, i) = a;
end

% prepare
% 80% for training, 20% for testing
% use attribute one & two
idx = randperm(row);  % reranke data
len = fix(row*0.8);
input_train(1:2,:)  = datas(1:2, idx(1:len));
group_train = datas(5, idx(1:len));
input_test(1:2, :)  = datas(1:2, idx(len+1:end));
group_test  = datas(5, idx(len+1:end));

% linear SVM training & testing
svm1 = svmtrain(input_train', group_train', 'Kernel_Function', 'linear','Showplot',true );
pause;
% estimated function
group_esm = svmclassify(svm1, input_test', 'Showplot', true);
hold on;
% plot points
plot(input_test(:,1),input_test(:,2),'bs','Markersize',12);
% accuracy
accuracy1 = sum(strcmp(group_esm,group_test))/row*100;            
hold off;  




















