% 清理工作空间
clear;
clc;
close all;

% 检查必要的工具箱
required_toolboxes = {'Parallel Computing Toolbox'};
missing_toolboxes = {};

for i = 1:length(required_toolboxes)
    if ~license('test', required_toolboxes{i})
        missing_toolboxes{end+1} = required_toolboxes{i};
    end
end

if ~isempty(missing_toolboxes)
    error('缺少以下工具箱:\n%s\n请安装这些工具箱后再运行程序。', ...
        strjoin(missing_toolboxes, '\n'));
end

% 检查GPU是否可用
if gpuDeviceCount == 0
    error('未检测到支持CUDA的GPU设备。请确保您的计算机有支持CUDA的GPU，并已安装正确的驱动程序。');
end

% 显示GPU信息
gpu = gpuDevice();
fprintf('使用的GPU设备: %s\n', gpu.Name);
fprintf('GPU总内存: %.2f GB\n', gpu.TotalMemory/1e9);
fprintf('CUDA版本: %.1f\n', gpu.ComputeCapability);

% 物理常数
c = 3e8;          % 光速
fc = 16.2e9;      % 载频 16.2GHz
lambda = c/fc;    % 波长
PRT = 4e-5;       % 脉冲重复时间 40us
Br = 100e6;       % 调频带宽100MHz
Kr = Br/PRT;      % 调频斜率
fs = 15e6;        % 采样频率15MHz
Nr = round(2 * PRT * fs);  % 距离向采样点数
Na = 2000;        % 雷达位置个数
theta = deg2rad(20);  % 波束宽度 20度（转换为弧度）

% 时间轴设置
length_label = Nr;
t_label = linspace(-PRT/2, length_label/fs-PRT/2, length_label);

% 成像区域
x_picture_total = [400, 700];  % 距离范围
y_picture_total = [-100, 100]; % 角度范围
y_radar = linspace(-0.6, 0.6, Na);  % 雷达位置
target_pos = [500, 0];  % 目标位置

% 网格划分
dr = 0.886 * c / (2 * Br);  % 距离向分辨率
dy = 0.886 * c * target_pos(1) / fc / (2 * 1.2);  % 方位向分辨率
dt_x = dr / 5;
dt_y = dy / 5;
N_x = round((x_picture_total(2) - x_picture_total(1)) / dt_x);
N_y = round((y_picture_total(2) - y_picture_total(1)) / dt_y);
[Y, X] = meshgrid(linspace(y_picture_total(1), y_picture_total(2), N_y), ...
                 linspace(x_picture_total(1), x_picture_total(2), N_x));

% 生成回波数据
function dechirp_matrix = generate_echo_data(t_label, PRT, fc, Kr, y_radar, target_pos, theta, c, length_label)
    % 发射信号生成
    phase_carry = exp(1j * 2 * pi * t_label * fc);
    sig_origin = (abs(t_label) <= PRT/2) .* exp(1j * pi * Kr * t_label.^2) .* phase_carry;
    
    % 初始化矩阵
    echo_matrix = zeros(Na, length_label);
    up_n = 10;  % 升采样因子
    dechirp_matrix = zeros(Na, length_label*up_n);
    
    % 生成回波矩阵
    for m = 1:Na
        y_m = y_radar(m);
        R = sqrt((y_m - target_pos(2))^2 + target_pos(1)^2);
        delay = 2 * R / c;
        
        % 计算方位角并检查是否在波束范围内
        phi = atan2(target_pos(2) - y_m, target_pos(1));
        if abs(phi) <= theta/2
            t_delay = t_label + delay;
            phase_carry_rx = exp(1j * 2 * pi * t_delay * fc);
            s_rx = (abs(t_delay) <= PRT/2) .* exp(1j * pi * Kr * t_delay.^2) .* phase_carry_rx;
            
            echo_matrix(m, :) = s_rx;
            % dechirp处理
            sig_rd = s_rx .* conj(sig_origin);
            
            % 升采样处理
            sig_rd_padded = zeros(1, length_label*up_n);
            sig_rd_padded(1:length_label/2) = sig_rd(1:length_label/2);
            sig_rd_padded(end-length_label/2+1:end) = sig_rd(length_label/2+1:end);
            
            SIG_rd = fft(sig_rd_padded);
            dechirp_matrix(m, :) = SIG_rd;
        end
    end
end

% CPU版本的BP成像
function img = bp_imaging_cpu(dechirp_data, y_radar, X, Y, lambda, fs, Kr, c, up_n, length_label)
    [N_y, N_x] = size(X);
    img = zeros(N_y, N_x);
    
    % BP成像
    for x_idx = 1:N_x
        if mod(x_idx, 50) == 0
            fprintf('CPU Processing column %d/%d...\n', x_idx, N_x);
        end
        
        for y_idx = 1:N_y
            pixel_x = X(y_idx, x_idx);
            pixel_y = Y(y_idx, x_idx);
            
            for ix = 1:Na
                RadarPosNow = y_radar(ix);
                R_radar = sqrt((pixel_x)^2 + (pixel_y - RadarPosNow)^2);
                N = round(abs(2 * R_radar * Kr / (c * fs / (up_n * length_label))));
                
                if N >= 1 && N <= length_label * up_n
                    sig = dechirp_data(ix, N);
                    img(y_idx, x_idx) = img(y_idx, x_idx) + sig * exp(-1j * 4 * pi * R_radar / lambda);
                end
            end
        end
    end
end

% GPU版本的BP成像
function img = bp_imaging_gpu(dechirp_data, y_radar, X, Y, lambda, fs, Kr, c, up_n, length_label)
    % 将数据转移到GPU
    dechirp_data_gpu = gpuArray(dechirp_data);
    X_gpu = gpuArray(X);
    Y_gpu = gpuArray(Y);
    y_radar_gpu = gpuArray(y_radar);
    
    [N_y, N_x] = size(X);
    img_gpu = gpuArray(zeros(N_y, N_x));
    
    % BP成像
    for x_idx = 1:N_x
        if mod(x_idx, 50) == 0
            fprintf('GPU Processing column %d/%d...\n', x_idx, N_x);
        end
        
        for y_idx = 1:N_y
            pixel_x = X_gpu(y_idx, x_idx);
            pixel_y = Y_gpu(y_idx, x_idx);
            
            for ix = 1:Na
                RadarPosNow = y_radar_gpu(ix);
                R_radar = sqrt((pixel_x)^2 + (pixel_y - RadarPosNow)^2);
                N = round(abs(2 * R_radar * Kr / (c * fs / (up_n * length_label))));
                
                if N >= 1 && N <= length_label * up_n
                    sig = dechirp_data_gpu(ix, N);
                    img_gpu(y_idx, x_idx) = img_gpu(y_idx, x_idx) + sig * exp(-1j * 4 * pi * R_radar / lambda);
                end
            end
        end
    end
    
    % 将结果转回CPU
    img = gather(img_gpu);
end

% 主程序
% 生成回波数据
fprintf('Generating radar echo data...\n');
dechirp_data = generate_echo_data(t_label, PRT, fc, Kr, y_radar, target_pos, theta, c, length_label);

% CPU BP成像
fprintf('\nStarting CPU BP imaging...\n');
tic;
img_cpu = bp_imaging_cpu(dechirp_data, y_radar, X, Y, lambda, fs, Kr, c, 10, length_label);
cpu_time = toc;
fprintf('CPU Processing time: %.2f seconds\n', cpu_time);

% GPU BP成像
fprintf('\nStarting GPU BP imaging...\n');
tic;
img_gpu = bp_imaging_gpu(dechirp_data, y_radar, X, Y, lambda, fs, Kr, c, 10, length_label);
gpu_time = toc;
fprintf('GPU Processing time: %.2f seconds\n', gpu_time);

% 计算加速比
speedup = cpu_time / gpu_time;
fprintf('\nSpeedup: %.2fx\n', speedup);

% 显示结果
figure('Position', [100, 100, 1200, 500]);

% CPU结果
subplot(1,2,1);
imagesc(x_picture_total, y_picture_total, 20*log10(abs(img_cpu)/max(abs(img_cpu(:)))), [-40 0]);
colorbar;
colormap('jet');
title('CPU BP 成像结果');
xlabel('距离向（米）');
ylabel('方位向（米）');
axis xy;

% GPU结果
subplot(1,2,2);
imagesc(x_picture_total, y_picture_total, 20*log10(abs(img_gpu)/max(abs(img_gpu(:)))), [-40 0]);
colorbar;
colormap('jet');
title('GPU BP 成像结果');
xlabel('距离向（米）');
ylabel('方位向（米）');
axis xy;

% 添加处理时间信息到标题
sgtitle(sprintf('BP成像对比 (CPU: %.2fs, GPU: %.2fs, 加速比: %.2fx)', ...
    cpu_time, gpu_time, speedup));
