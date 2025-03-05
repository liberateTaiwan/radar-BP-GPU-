import numpy as np
import cupy as cp
import numba
from numba import cuda
import matplotlib.pyplot as plt
import time

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 物理常数
c = 3e8  # 光速
fc = 16.2e9  # 载频 16.2GHz
lambda_ = c / fc  # 波长
PRT = 4e-5  # 脉冲重复时间 40us
Br = 100e6  # 调频带宽100MHz
Kr = Br / PRT  # 调频斜率
fs = 15e6  # 采样频率15MHz
Nr = int(2 * PRT * fs)  # 距离向采样点数
Na = 2000  # 雷达位置个数
theta = np.radians(20)  # 波束宽度 20度（转换为弧度）

# 时间轴设置
length_label = Nr
t_label = np.linspace(-PRT/2, length_label/fs-PRT/2, length_label)

# 成像区域
x_picture_total = [400, 700]  # 距离范围
y_picture_total = [-100, 100]  # 角度范围
y_radar = np.linspace(-0.6, 0.6, Na)  # 雷达位置
target_pos = np.array([500, 0])  # 目标位置

# 网格划分
dr = 0.886 * c / (2 * Br)  # 距离向分辨率
dy = 0.886 * c * target_pos[0] / fc / (2 * 1.2)  # 方位向分辨率
dt_x = dr / 5
dt_y = dy / 5
N_x = round((x_picture_total[1] - x_picture_total[0]) / dt_x)
N_y = round((y_picture_total[1] - y_picture_total[0]) / dt_y)
y_grid = np.linspace(y_picture_total[0], y_picture_total[1], N_y)
x_grid = np.linspace(x_picture_total[0], x_picture_total[1], N_x)
Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')

def generate_echo_data():
    # 发射信号生成
    phase_carry = np.exp(1j * 2 * np.pi * t_label * fc)
    sig_origin = np.where(np.abs(t_label) <= PRT/2, 1, 0) * np.exp(1j * np.pi * Kr * t_label**2) * phase_carry
    
    # 初始化矩阵
    echo_matrix = np.zeros((Na, length_label), dtype=np.complex64)
    up_n = 10  # 升采样因子
    dechirp_matrix = np.zeros((Na, length_label*up_n), dtype=np.complex64)
    
    # 生成回波矩阵
    for m in range(Na):
        y_m = y_radar[m]
        R = np.sqrt((y_m - target_pos[1])**2 + target_pos[0]**2)
        delay = 2 * R / c
        
        # 计算方位角并检查是否在波束范围内
        phi = np.arctan2(target_pos[1] - y_m, target_pos[0])
        if np.abs(phi) <= theta/2:
            t_delay = t_label + delay
            phase_carry_rx = np.exp(1j * 2 * np.pi * t_delay * fc)
            s_rx = np.where(np.abs(t_delay) <= PRT/2, 1, 0) * np.exp(1j * np.pi * Kr * t_delay**2) * phase_carry_rx
            
            echo_matrix[m, :] = s_rx
            # dechirp处理
            sig_rd = s_rx * np.conj(sig_origin)
            
            # 升采样处理 - 修改补零方式
            sig_rd_padded = np.zeros(length_label*up_n, dtype=np.complex64)
            sig_rd_padded[:length_label//2] = sig_rd[:length_label//2]
            sig_rd_padded[-length_label//2:] = sig_rd[length_label//2:]
            
            SIG_rd = np.fft.fft(sig_rd_padded)
            dechirp_matrix[m, :] = SIG_rd
    
    return dechirp_matrix

# 生成实际的雷达数据
print("Generating radar echo data...")
dechirp_data = generate_echo_data()
dechirp_real_cpu = np.real(dechirp_data).astype(np.float32)
dechirp_imag_cpu = np.imag(dechirp_data).astype(np.float32)

# 修改BP成像算法中的距离计算
def calculate_range_index(R, fs, up_n, length_label, Kr):
    return int(abs(round(2 * R * Kr / (c * fs / (up_n * length_label)))))

# CPU 版本 BP 成像
def bp_imaging_cpu():
    img = np.zeros((N_y, N_x), dtype=np.complex64)
    up_n = 10

    for x_idx in range(N_x):
        for y_idx in range(N_y):
            if (x_idx % 50 == 0 and y_idx == 0):
                print(f"Processing column {x_idx}/{N_x}...")

            pixel_x = X[y_idx, x_idx]
            pixel_y = Y[y_idx, x_idx]

            for ix in range(Na):
                RadarPosNow = y_radar[ix]
                R_radar = np.sqrt((pixel_x - 0) ** 2 + (pixel_y - RadarPosNow) ** 2)
                N = calculate_range_index(R_radar, fs, up_n, length_label, Kr)

                if 0 <= N < length_label * up_n:
                    sig = complex(dechirp_real_cpu[ix, N], dechirp_imag_cpu[ix, N])
                    img[y_idx, x_idx] += sig * np.exp(-1j * 4 * np.pi * R_radar / lambda_)

    return np.abs(img)

# CUDA 设备内核：BP 成像
@cuda.jit
def bpImagingKernel(img_real, img_imag, dechirp_real, dechirp_imag, y_radar, X, Y, up_n, lambda_, fs, Kr, c):
    x_idx, y_idx = cuda.grid(2)
    width, height = img_real.shape
    
    if x_idx >= width or y_idx >= height:
        return

    pixel_x = X[y_idx, x_idx]
    pixel_y = Y[y_idx, x_idx]

    img_real_val = 0.0
    img_imag_val = 0.0

    for ix in range(Na):
        RadarPosNow = y_radar[ix]
        R_radar = ((pixel_x - 0) ** 2 + (pixel_y - RadarPosNow) ** 2) ** 0.5
        N = int(abs(round(2 * R_radar * Kr / (c * fs / (up_n * width)))))

        if 0 <= N < width * up_n:
            sig_real = dechirp_real[ix, N]
            sig_imag = dechirp_imag[ix, N]

            phase = -4.0 * np.pi * R_radar / lambda_
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)

            img_real_val += sig_real * cos_phase - sig_imag * sin_phase
            img_imag_val += sig_real * sin_phase + sig_imag * cos_phase

    img_real[y_idx, x_idx] = img_real_val
    img_imag[y_idx, x_idx] = img_imag_val

# CUDA 版本 BP 计算
def bp_imaging_gpu():
    print("Starting CUDA computation...")
    img_real = cp.zeros((N_y, N_x), dtype=cp.float32)
    img_imag = cp.zeros((N_y, N_x), dtype=cp.float32)
    d_y_radar = cp.array(y_radar, dtype=cp.float32)
    d_X = cp.array(X, dtype=cp.float32)
    d_Y = cp.array(Y, dtype=cp.float32)
    dechirp_real = cp.array(dechirp_real_cpu, dtype=cp.float32)
    dechirp_imag = cp.array(dechirp_imag_cpu, dtype=cp.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (N_x + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N_y + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print("Launching CUDA kernel...")
    try:
        bpImagingKernel[blocks_per_grid, threads_per_block](
            img_real, img_imag, dechirp_real, dechirp_imag, 
            d_y_radar, d_X, d_Y, 10, lambda_, fs, Kr, c
        )
        print("CUDA kernel finished!")
    except Exception as e:
        print(f"CUDA Error: {e}")

    return cp.sqrt(img_real ** 2 + img_imag ** 2).get()

# 计算 CPU 和 GPU 时间并比较结果
print("Starting CPU computation...")
start_cpu = time.time()
img_cpu = bp_imaging_cpu()
time_cpu = time.time() - start_cpu

print("\nStarting GPU computation...")
start_gpu = cp.cuda.Event()
end_gpu = cp.cuda.Event()
start_gpu.record()
img_gpu = bp_imaging_gpu()
end_gpu.record()
end_gpu.synchronize()
time_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000

# 打印性能比较
print(f"\nCPU computation time: {time_cpu:.4f} seconds")
print(f"GPU computation time: {time_gpu:.4f} seconds")
print(f"Speedup: {time_cpu / time_gpu:.2f}x")

# 绘制结果
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.imshow(20*np.log10(np.abs(img_cpu)/np.max(np.abs(img_cpu))), 
           extent=[x_picture_total[0], x_picture_total[1], 
                  y_picture_total[0], y_picture_total[1]], 
           aspect='auto',
           vmin=-40)
plt.colorbar(label='dB')
plt.title('CPU BP 成像结果')
plt.xlabel('距离向（米）')
plt.ylabel('方位向（米）')

plt.subplot(122)
plt.imshow(20*np.log10(np.abs(img_gpu)/np.max(np.abs(img_gpu))), 
           extent=[x_picture_total[0], x_picture_total[1], 
                  y_picture_total[0], y_picture_total[1]], 
           aspect='auto',
           vmin=-40)
plt.colorbar(label='dB')
plt.title('GPU BP 成像结果')
plt.xlabel('距离向（米）')
plt.ylabel('方位向（米）')

plt.tight_layout()
plt.show()