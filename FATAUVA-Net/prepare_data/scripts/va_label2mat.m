clear;
labelpath = 'F:\FATAUVA-Net\prepare_data\AFEW-VA\crop\labels.txt';

f=fopen(labelpath,'r');

i=1;
labels=zeros(21118, 2);
% ∂¡»°label÷µ
while feof(f)==0
    line=regexp(fgetl(f),' +', 'split');
    for j=2:length(line)
        labels(i,j-1)=str2double(line{j});
    end
    i=i+1;
end

save ..\AFEW-VA\crop\labels labels