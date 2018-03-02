clear;

% 文件路径
all = 'F:\FATAUVA-Net\prepare_data\AFEW-VA\crop\data.txt';
train_data = 'F:\FATAUVA-Net\prepare_data\AFEW-VA\crop\train_data.txt';
test_data = 'F:\FATAUVA-Net\prepare_data\AFEW-VA\crop\test_data.txt';

[filenames,classes]=textread(all,'%s %d');

num_train = 17650;
num_test = 3468;

f1=fopen(train_data,'w');
f2=fopen(test_data,'w');

% 写入不同文件
for i=1:num_train+num_test
    if i<=num_train
        idx = min(strfind(filenames{i},'\'));
        fullname = filenames{i};
        filename = fullname(idx+1:length(fullname));
        fprintf(f1, '%s %d\n', filename, classes(i));
    end
    if (num_train)<i && i<=(num_train+num_test)
        idx = min(strfind(filenames{i},'\'));
        fullname = filenames{i};
        filename = fullname(idx+1:length(fullname));
        fprintf(f2, '%s %d\n', filename, classes(i));
    end      
end

fclose(f1);
fclose(f2);
