clc;clear;close all

% Path/to/results
root = "../results/dvToF_norm_T_weight";
scenes = {'bathroom/0', 'bathroom/7', 'white-room/7', 'white-room/0', 'breakfast/0', 'contemporary-bathroom/7', 'pavilion/1', 'breakfast/1', 'contemporary-bathroom/2', 'pavilion/6'};

for i = 1:length(scenes)
    scene = scenes{i};

    % makedir
    dir_path = fullfile(root, 'depth_mats', scene);
    if ~exist(dir_path, 'dir') 
        mkdir(dir_path);  
    end
    disp(dir_path)
        
    for k = 2:250
        noise_raw_file = fullfile(root, 'corr', scene, [num2str(k), '.mat']);
        noise_depth_file = fullfile(root, 'depth_mats', scene, [num2str(k), '.mat']);

        load(noise_raw_file)
        freqVec = [40, 1e2 / 3.3, 1e2 / 1.7] * 1e6;
        maxd = 10;
        nt = 5000;
        nf = numel(freqVec);
        h = corr_imgs; 
        h0mat = h(1:nf,:,:); %cos
        h90mat = h(nf+1:end,:,:); %sin
        corr_imgs = h0mat + 1i*h90mat;
        phase_imgs = angle(corr_imgs);
        for fi = 1:nf
            tmp = squeeze(phase_imgs(fi,:,:)<0);
            phase_imgs(fi,tmp) = 2*pi + phase_imgs(fi,tmp);
        end
        corr_imgs = cat(1,h0mat,h90mat);
        delayVec = linspace(0,2*maxd,nt);
        depths = PhaseImgs2Depths(freqVec, phase_imgs, delayVec/2);

        save(noise_depth_file)
    end
end
