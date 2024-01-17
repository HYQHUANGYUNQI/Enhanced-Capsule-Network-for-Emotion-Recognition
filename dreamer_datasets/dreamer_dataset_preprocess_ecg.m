clear;

data_loader = load('./DREAMER.mat'); 

for i = 1:23
    saldir = '../data/dreamer_data_ecg/stimuli(m,2)/';
    data_stimuli = data_loader.DREAMER.Data{1,i}.ECG.stimuli;  
    for j = 1:18
        data_stimuli{j,1} = data_loader.DREAMER.Data{1,i}.ECG.stimuli{j,1};   
        trial_name = ['data_stimuli',num2str(j)];
        eval([trial_name,'=data_stimuli{j,1}']);  
    end
    
    savepath = [saldir,'ecg', num2str(i),'_stimuli'];
    save(savepath,'data_stimuli1','data_stimuli2','data_stimuli3','data_stimuli4','data_stimuli5','data_stimuli6','data_stimuli7','data_stimuli8','data_stimuli9','data_stimuli10','data_stimuli11','data_stimuli12','data_stimuli13','data_stimuli14','data_stimuli15','data_stimuli16','data_stimuli17','data_stimuli18');
end


for i = 1:23
    saldir = '../data/dreamer_data_ecg/baseline(m,2)/';
    data_baseline = data_loader.DREAMER.Data{1,i}.ECG.baseline; 
    for j = 1:18
        data_baseline{j,1} = data_loader.DREAMER.Data{1,i}.ECG.baseline{j,1};  
        trial_name = ['data_baseline',num2str(j)];
        eval([trial_name,'=data_baseline{j,1}']);   
    end
    
    savepath = [saldir,'ecg',num2str(i),'_baseline'];
    save(savepath,'data_baseline1','data_baseline2','data_baseline3','data_baseline4','data_baseline5','data_baseline6','data_baseline7','data_baseline8','data_baseline9','data_baseline10','data_baseline11','data_baseline12','data_baseline13','data_baseline14','data_baseline15','data_baseline16','data_baseline17','data_baseline18');
end

for i = 1:23
    saldir = '../data/dreamer_data_ecg/label_arousal(18,1)/';
    data_label = data_loader.DREAMER.Data{1,i}.ScoreArousal;  
    savepath = [saldir,num2str(i),'_arousal','_label'];  
    save(savepath,'data_label');  
end
