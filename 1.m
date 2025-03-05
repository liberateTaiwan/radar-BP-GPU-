% clc;
% close all;
% clear;
%% 参数设置 
c = 3e8;               % 光速 
fc =16.2e9;            % 载频 16.2GHz
lambda = c/fc;         % 波长 
PRT = 4e-5;            % 脉冲重复时间 40us 
Br = 100e6;            % 调频带宽100MHz
Kr = Br / PRT;         % 调频斜率  
fs =15e6;              % 采样频率15MHz  
Nr = 2*PRT*fs;         % 距离向采样点数
Na = 2000;              % 雷达位置个数（PRT个数）（方向位采样点数）(要满足采样定理，太少了）      
theta =20 * pi / 180;  % 波束宽度 20度，转换为弧度
%% 模型建立
length_label = round(2*PRT*fs);  %横轴长度
t_label = linspace(-PRT/2,length_label/fs-PRT/2,length_label);  %时间横轴
x_picture_total = [400,700];  %成像区域距离范围
y_picture_total = [-100,100];  %成像区域角度范围
y_radar = linspace(-0.6, 0.6, Na); % 雷达在方位向的位置
%成像区域及目标点设置
target_pos = [500 0];%目标坐标
dr = 0.886*c/(2*Br);  %距离向分辨率
fprintf('理论距离向分辨率：%.4f 米\n', dr);
dy =0.886*c*target_pos(1)/fc/(2*1.2);% 方位向分辨率
fprintf('理论方位向分辨率：%.4f 米\n', dy);
%% 回波构建和去斜
phase_carry = exp(1i*pi*2*t_label*fc);%载频
sig_origin = rectpuls(t_label,PRT).*exp(1i*pi*Kr*t_label.^2).*phase_carry;%发射信号
%初始化回波矩阵
echo_matrix = zeros(Na, length_label);%每个采样点上的回波信息
dechirp_matrix = zeros(Na, 2*PRT*fs*10);

%生成回波矩阵
for m = 1:Na
    %当前雷达位置
    y_m = y_radar(m);
    %计算雷达与目标距离
    R = sqrt((y_m - target_pos(2))^2 + target_pos(1)^2);
    %计算延时  
    delay = 2 * R / c;
    % 判断目标是否在雷达波束内
    phi = atan2(target_pos(2) - y_m, target_pos(1)); % 目标的方位角
    if abs(phi) <= theta / 2
        t_delay = t_label + delay;
        
        phase_carry_rx = exp(1i*pi*2*t_delay*fc);%载频

        s_rx = rectpuls(t_delay,PRT).*exp(1i*pi*Kr*t_delay.^2).*phase_carry_rx;%回波
        % 将接收信号存入回波矩阵
        echo_matrix(m, :) = s_rx;

         % 与发射信号进行共轭相乘（dechirp）
        sig_rd = s_rx .* conj(sig_origin); 
        up_n = 10;%升采样因子
        %升采样 时域补零 频域插值
        sig_rd = [sig_rd(:,1:length_label/2),zeros(1,(up_n-1)*length_label),sig_rd(:,(length_label/2+1):length_label)];%在时域补零
        SIG_rd = fft(sig_rd,[],2);%fft转化为频域为什么bp成像叫时域成像算法
        %存入矩阵中
        dechirp_matrix(m, :) = SIG_rd;
    end
end

% 绘制单目标回波矩阵图像
figure;
imagesc(real(echo_matrix));
title('单目标回波矩阵（实部）');
xlabel('距离向');
ylabel('prt');
colorbar;

%绘制去斜后回波矩阵
r_ref = (0:(up_n*length_label-1)) * (fs / (up_n*length_label)) * c / (2*Kr);
figure;
imagesc(r_ref, y_radar,abs(dechirp_matrix));  % 使用实部作图
title('dechirp后回波矩阵的频谱图');
xlabel('距离向');
ylabel('雷达位置');
colorbar;


%% 网格划分
dt_x = dr/5;
dt_y = dy/5;
N_x = round((x_picture_total(2)-x_picture_total(1))/dt_x);
N_y = round((y_picture_total(2)-y_picture_total(1))/dt_y);
y = linspace(y_picture_total(1),y_picture_total(2),N_y);
x = linspace(x_picture_total(1),x_picture_total(2),N_x);
Y = y'*ones(1,N_x);
X = ones(N_y,1)*x;

R_radar = zeros(N_y,N_x);
img = zeros(size(Y));  
%% bp成像
for ix = 1:Na
    RadarPosNow = y_radar(ix); % 当前雷达位置
    % 计算雷达与成像区域中每个点的距离
    R_radar = sqrt((X- 0).^2 + (Y - RadarPosNow).^2);  % 计算每个成像点与雷达的距离
    % 将距离转换为对应的采样点数
    N = abs(round(R_radar/((fs/(up_n*length_label))*c/Kr/2)));
    sig_rdta = dechirp_matrix(ix, :);
    img = img +  sig_rdta(N) .* exp( - 1j * 4 * pi .* R_radar / lambda );  % 相位补偿
end

%% 绘制成像结果
figure;
x = linspace(x_picture_total(1),x_picture_total(2),N_x);
y = linspace(y_picture_total(1),y_picture_total(2),N_y);
imagesc(x, y,abs(img)); % 绘制振幅图像 
title('后向投影成像结果');
xlabel('距离向（米）');
ylabel('雷达方位向');
colorbar;


