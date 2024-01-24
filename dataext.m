% Load data
fname = 'D:\Wifall\cfr\cfr_11.txt'; % Update with your filename
data = load(fname);
                                                
% 执行PCA
[coeff,score,~,~,explained] = pca(data);

% 获得第一主成分
pc1 = score(:,1);

% 计算第一阶差分
phase_info = diff(pc1);

% Plot phase & freq info
figure;
subplot(2,1,1);
plot(phase_info);
title('振幅的第一差分');
hold on; % 保持当前图形，以便在上面叠加

threshold = 0.45; 
% Compute start indices of actions
start_indices = find(abs(phase_info) > threshold);
start_indices = start_indices + 1; % Add 2 because diff reduces the vector length by 1 for each application

% Display start_indices
%disp('Start indices of actions are: ');
%disp(start_indices);

% 阈值
threshold = 320;
%第一波阈值切割
% 初始化空的聚类数组
clusters = [];
maxL = -1;
minL = 1000000000;
% 设置开始索引
start_index = start_indices(1);

% 遍历数组中的每个元素
for i = 2:length(start_indices)
    % 检查当前元素与前一个元素之间的差值
    if start_indices(i) - start_indices(i-1) > threshold
        % 记录当前聚类的结束索引，并开始一个新的聚类
        %寻找最大间隔
        if start_indices(i) - start_indices(i-1) > maxL
            maxL = start_indices(i) - start_indices(i-1);
        end
        %寻找最小间隔
        if start_indices(i) - start_indices(i-1) < minL
            minL = start_indices(i) - start_indices(i-1);
        end
        clusters = [clusters; start_index];
        clusters = [clusters; start_indices(i - 1)];
        start_index = start_indices(i);
    end
end
% 添加最后一组
clusters = [clusters; start_index];
clusters = [clusters; start_indices(length(start_indices))];

%第二波阈值切割
new_threshold = (minL + maxL) / 2;
%disp(new_threshold);
% 初始化空的聚类数组
clusters1 = [];
% 设置开始索引
start_index1 = start_indices(1);

% 遍历数组中的每个元素
for i = 2:length(start_indices)
    % 检查当前元素与前一个元素之间的差值
    if start_indices(i) - start_indices(i-1) > threshold
        % 记录当前聚类的结束索引，并开始一个新的聚
        clusters1 = [clusters1; start_index1];
        clusters1 = [clusters1; start_indices(i - 1)];
        start_index1 = start_indices(i); 
    end
end
% 添加最后一组
clusters1 = [clusters; start_index1];
clusters1 = [clusters; start_indices(length(start_indices))];

clusters1 = clusters1(1:end-1)
% 在特定的索引处画红色的直线
yLimits = ylim; % 获取当前y轴的限制
for i = 1:length(clusters1)
    index = clusters1(i);
    %disp(clusters1(i));
    line([index, index], yLimits, 'Color', 'red', 'LineWidth', 2); % 画一条垂直的红色直线
end
%disp(length(clusters1));
%disp(data(clusters1(1):clusters1(2), :));
hold off; % 取消保持，完成绘图

%标签
labelArray = {'Walk', 'Fall', 'Stand', 'Sit', 'Stand','Fall', 'Stand', 'Sit', 'Stand', 'Fall','Stand', 'Sit', 'Stand', 'Walk' };;

% 提取奇数位和偶数位
startColumn = clusters1(1:2:end)';
endColumn = clusters1(2:2:end)';
disp(length(startColumn));
disp(length(labelArray));
% 创建一个表格
myTable = table(startColumn, endColumn, 'VariableNames', {'Start', 'End'});

% 使用addvars函数增加新列
myTable = addvars(myTable, labelArray, 'NewVariableNames', 'Label');

% 指定要保存的文件名
%filename = 'cfr_21.csv';

% 使用writetable将表格保存到CSV文件
%writetable(myTable, filename);


