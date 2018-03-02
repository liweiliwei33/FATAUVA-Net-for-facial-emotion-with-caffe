clear;
attrlabelpath = 'E:\CelebA\Anno\list_attr_celeba.txt';
face_attrs = {'Attractive','Male','No_Beard','Young'};
eyebrows_attrs = {'Arched_Eyebrows','Bushy_Eyebrows'};
eyes_attrs = {'Eyeglasses','Narrow_Eyes'};
mouth_attrs = {'Mouth_Slightly_Open','Smiling'};
attr_names = [face_attrs, eyebrows_attrs, eyes_attrs, mouth_attrs];

f=fopen(attrlabelpath,'r');
num = str2double(fgetl(f));     % 读取第一行，总数
attrs = regexp(fgetl(f),' ','split');       % 第二行，属性名
[bools, indexs] = ismember(attr_names, attrs);      % 找到需要的属性下标

i=1;
all_labels=zeros(num, 40);
% 读取label值
while feof(f)==0
    line=regexp(fgetl(f),' +', 'split');
    for j=2:length(line)
        all_labels(i,j-1)=str2double(line{j});
    end
    i=i+1;
end
all_labels(find(all_labels==-1))=0;
labels=all_labels(:,indexs);
save ..\labels labels