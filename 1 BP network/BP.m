clear;

% ------------ UCI Dataset Attribute Information -------------------------
%  name: iris
%   1. sepal length in cm
%   2. sepal width in cm
%   3. petal length in cm
%   4. petal width in cm
%   5. class: 
%      -- Iris Setosa       ->output 1
%      -- Iris Versicolour  ->output 2
%      -- Iris Virginica    ->output 3
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
            a = 0;
        case('Iris-versicolor')
            a = 1;
        case('Iris-virginica')
            a = 2;
    end
    datas(col, i) = a;
end

% prepare
% 80% for training, 20% for testing
idx = randperm(row);  % reranke data
len = fix(row*0.8);
input_train(1:4,:)  = datas(1:4, idx(1:len));
output_train = datas(5, idx(1:len));
input_test(1:4, :)  = datas(1:4, idx(len+1:end));
output_test  = datas(5, idx(len+1:end));


% len = fix(row/4.0);
% input_train  = zeros(4, row - 4*len);
% output_train = zeros(1, row - 4*len);
% input_test  = zeros(4, len);
% output_test = zeros(1, len);
% for i = 1:row
%     if mod(i,4) == 0
%         input_train(1:4, i) = datas(1:4, i);
%         output_train(1,  i) = datas(5, i);
%     else
%         input_test(1:4, i) = datas(1:4, i);
%         output_test(1,  i) = datas(5, i);
%     end
% end

% BP network
% P: input  [R*Q1]
% T: output [SN*Q2]
% one hidden layer, number of units = 10
% cost function
% threshold function , 'tangis' for hidden layer, 'purelin' for outpur
net = newff(input_train, output_train, [10, 4], {'logsig','logsig', 'purelin'}, 'trainlm');
net.trainParam.goal = 1e-5;
net.trainParam.epochs = 300;
% effects of msereg are isolated from early stopping
net.divideFcn = '';

% training
net = train(net, input_train, output_train);

% test
R = sim(net, input_test) - output_test;
Y2 = sim(net, input_test);
