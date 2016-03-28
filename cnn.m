clear all;
Files = dir('00\*.jpg');
LengthFiles = length(Files);
inputs = [];
for i = 1:LengthFiles;
    img = imread(strcat('00\',Files(i).name));
    img = reshape(img, [1600,1]);
    inputs = [inputs img];
end

inputs = im2double(inputs);

targets = [];
%0
for i = 1:16;
    s = sparse(1,1,1,33,1);
    targets = [targets, s];
end

%1
for i = 1:12;
    s = sparse(2,1,1,33,1);
    targets = [targets, s];
end

%2
for i = 1:40;
    s = sparse(3,1,1,33,1);
    targets = [targets, s];
end

%3
for i = 1:19;
    s = sparse(4,1,1,33,1);
    targets = [targets, s];
end

%4
for i = 1:36;
    s = sparse(5,1,1,33,1);
    targets = [targets, s];
end

%5
for i = 1:19;
    s = sparse(6,1,1,33,1);
    targets = [targets, s];
end

%6
for i = 1:36;
    s = sparse(7,1,1,33,1);
    targets = [targets, s];
end

%7
for i = 1:16;
    s = sparse(8,1,1,33,1);
    targets = [targets, s];
end

%8
for i = 1:17;
    s = sparse(9,1,1,33,1);
    targets = [targets, s];
end

%9
for i = 1:19;
    s = sparse(10,1,1,33,1);
    targets = [targets, s];
end

%A
for i = 1:20;
    s = sparse(11,1,1,33,1);
    targets = [targets, s];
end

%B
for i = 1:36;
    s = sparse(12,1,1,33,1);
    targets = [targets, s];
end

%C
for i = 1:36;
    s = sparse(13,1,1,33,1);
    targets = [targets, s];
end

%D
for i = 1:22;
    s = sparse(14,1,1,33,1);
    targets = [targets, s];
end

%E
for i = 1:35;
    s = sparse(15,1,1,33,1);
    targets = [targets, s];
end

%F
for i = 1:41;
    s = sparse(16,1,1,33,1);
    targets = [targets, s];
end

%G
for i = 1:31;
    s = sparse(17,1,1,33,1);
    targets = [targets, s];
end

%H
for i = 1:14;
    s = sparse(18,1,1,33,1);
    targets = [targets, s];
end

%J
for i = 1:15;
    s = sparse(19,1,1,33,1);
    targets = [targets, s];
end

%K
for i = 1:26;
    s = sparse(20,1,1,33,1);
    targets = [targets, s];
end

%L
for i = 1:18;
    s = sparse(21,1,1,33,1);
    targets = [targets, s];
end

%M
for i = 1:12;
    s = sparse(22,1,1,33,1);
    targets = [targets, s];
end

%N
for i = 1:12;
    s = sparse(23,1,1,33,1);
    targets = [targets, s];
end

%P
for i = 1:14;
    s = sparse(24,1,1,33,1);
    targets = [targets, s];
end

%R
for i = 1:21;
    s = sparse(25,1,1,33,1);
    targets = [targets, s];
end

%S
for i = 1:27;
    s = sparse(26,1,1,33,1);
    targets = [targets, s];
end

%T
for i = 1:18;
    s = sparse(27,1,1,33,1);
    targets = [targets, s];
end

%U
for i = 1:20;
    s = sparse(28,1,1,33,1);
    targets = [targets, s];
end

%V
for i = 1:18;
    s = sparse(29,1,1,33,1);
    targets = [targets, s];
end

%W
for i = 1:21;
    s = sparse(30,1,1,33,1);
    targets = [targets, s];
end

%X
for i = 1:13;
    s = sparse(31,1,1,33,1);
    targets = [targets, s];
end

%Y
for i = 1:24;
    s = sparse(32,1,1,33,1);
    targets = [targets, s];
end

%Z
for i = 1:21;
    s = sparse(33,1,1,33,1);
    targets = [targets, s];
end

targets = full(targets);




% Create a Pattern Recognition Network
hiddenLayerSize = 80;
net = patternnet(hiddenLayerSize);
net = configure(net,inputs,targets);


% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
%net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 20/100;


% Train the Network
net = init(net);
%net.trainFcn = 'traingdm';
[net,tr] = train(net,inputs,targets);

% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
% figure, plotperform(tr)
% figure, plottrainstate(tr)
 figure, plotconfusion(targets,outputs)
% figure, ploterrhist(errors)
