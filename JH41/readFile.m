file_name = 'D:\LEARN\4th SEMESTER\TensorFlow\JH41\mris.txt';
list = textread(file_name, '%s');
load ('list_file2.mat');
pidx = 1;
list2 = cell(0);
for i=1:length(list)
    temp = list{i,1};
    underscore_index = strfind(temp, '_');
    list{i,1} = temp(1:(underscore_index(3)-1));
end

stt = cell(0);
for i=1:length(list)
    for j=1:length(list_file)
        if strcmp(list{i,1}, list_file(j).name)
            stt{i,1} = list_file(j).status;
            % convert -1 to 0
            if list_file(j).status == -1
                stt{i,1} = 0;
            end
        end
    end
end

fileW = fopen('D:\LEARN\4th SEMESTER\TensorFlow\JH41\mris_label.txt','w');
[nrows,ncols] = size(stt);
formatSpec = '%d\n';
for row = 1:nrows
    fprintf(fileW,formatSpec,stt{row,:});
end