#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"

// �����Щ����
#define MATPLOTLIB_HEADER_ONLY
#ifdef _MSC_VER
#pragma warning(disable: 4244)
#pragma warning(disable: 4267)
#endif

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cufft.h>
#include "matplotlibcpp.h"

// ���� M_PI�����δ���壩
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace plt = matplotlibcpp;

// ������
const double c = 3e8;  // ����
const double fc = 16.2e9;  // ��Ƶ 16.2GHz
const double lambda = c / fc;  // ����
const double PRT = 4e-5;  // �����ظ�ʱ�� 40us
const double Br = 100e6;  // ��Ƶ����100MHz
const double Kr = Br / PRT;  // ��Ƶб��
const double fs = 15e6;  // ����Ƶ��15MHz
const int Nr = static_cast<int>(2 * PRT * fs);  // �������������
const int Na = 2000;  // �״�λ�ø���
const double theta = 20.0 * M_PI / 180.0;  // ������ȣ����ȣ�

// �����������
const double x_picture_min = 400.0;
const double x_picture_max = 700.0;
const double y_picture_min = -100.0;
const double y_picture_max = 100.0;

// Ŀ��λ��
const std::vector<double> target_pos = {500.0, 0.0};

// ���񻮷�
const double dr = 0.886 * c / (2 * Br);  // ������ֱ���
const double dy = 0.886 * c * target_pos[0] / fc / (2 * 1.2);  // ��λ��ֱ���
const double dt_x = dr / 5;
const double dt_y = dy / 5;
const int N_x = round((x_picture_max - x_picture_min) / dt_x);
const int N_y = round((y_picture_max - y_picture_min) / dt_y);

// CUDA�˺�������
__global__ void bpImagingKernel(float* img_real, float* img_imag,
                               const float* dechirp_real, const float* dechirp_imag,
                               const float* y_radar, const float* X, const float* Y,
                               int up_n, float lambda, float fs, float Kr, float c,
                               int Na, int N_x, int N_y);

// ���ɻز�����
std::vector<std::complex<float>> generate_echo_data() {
    // ʱ��������
    std::vector<double> t_label(Nr);
    for (int i = 0; i < Nr; ++i) {
        t_label[i] = -PRT/2 + i * (PRT/Nr);
    }

    // �״�λ��
    std::vector<double> y_radar(Na);
    for (int i = 0; i < Na; ++i) {
        y_radar[i] = -0.6 + i * 1.2 / (Na-1);
    }

    // ��ʼ������
    const int up_n = 10;  // ����������
    std::vector<std::complex<float>> dechirp_matrix(Na * Nr * up_n);

    // �����ź�����
    std::vector<std::complex<float>> sig_origin(Nr);
    for (int i = 0; i < Nr; ++i) {
        double t = t_label[i];
        if (std::abs(t) <= PRT/2) {
            double phase = 2 * M_PI * t * fc + M_PI * Kr * t * t;
            sig_origin[i] = std::complex<float>(std::cos(phase), std::sin(phase));
        }
    }

    // ���ɻز�����
    #pragma omp parallel for
    for (int m = 0; m < Na; ++m) {
        double y_m = y_radar[m];
        double R = std::sqrt(std::pow(y_m - target_pos[1], 2) + std::pow(target_pos[0], 2));
        double delay = 2 * R / c;
        
        // ���㷽λ��
        double phi = std::atan2(target_pos[1] - y_m, target_pos[0]);
        
        if (std::abs(phi) <= theta/2) {
            std::vector<std::complex<float>> s_rx(Nr);
            for (int i = 0; i < Nr; ++i) {
                double t = t_label[i] + delay;
                if (std::abs(t) <= PRT/2) {
                    double phase = 2 * M_PI * t * fc + M_PI * Kr * t * t;
                    s_rx[i] = std::complex<float>(std::cos(phase), std::sin(phase));
                }
            }

            // dechirp����
            std::vector<std::complex<float>> sig_rd(Nr);
            for (int i = 0; i < Nr; ++i) {
                sig_rd[i] = s_rx[i] * std::conj(sig_origin[i]);
            }

            // ����������
            std::vector<std::complex<float>> sig_rd_padded(Nr * up_n);
            for (int i = 0; i < Nr/2; ++i) {
                sig_rd_padded[i] = sig_rd[i];
                sig_rd_padded[Nr * up_n - Nr/2 + i] = sig_rd[Nr/2 + i];
            }

            // FFT
            // ������Ҫʹ��FFTW��cuFFT��ʵ��FFT
            // Ϊ��ʾ��������ʡ��FFTʵ��
            
            // �洢���
            for (int i = 0; i < Nr * up_n; ++i) {
                dechirp_matrix[m * Nr * up_n + i] = sig_rd_padded[i];
            }
        }
    }

    return dechirp_matrix;
}

// BP���� CPU�汾
std::vector<float> bp_imaging_cpu(const std::vector<std::complex<float>>& dechirp_data) {
    std::vector<float> img(N_y * N_x, 0.0f);
    const int up_n = 10;

    // ���������
    std::vector<float> x_grid(N_x), y_grid(N_y);
    for (int i = 0; i < N_x; ++i) {
        x_grid[i] = x_picture_min + i * dt_x;
    }
    for (int i = 0; i < N_y; ++i) {
        y_grid[i] = y_picture_min + i * dt_y;
    }

    #pragma omp parallel for collapse(2)
    for (int y_idx = 0; y_idx < N_y; ++y_idx) {
        for (int x_idx = 0; x_idx < N_x; ++x_idx) {
            float pixel_x = x_grid[x_idx];
            float pixel_y = y_grid[y_idx];
            
            std::complex<float> pixel_value(0.0f, 0.0f);
            
            for (int ix = 0; ix < Na; ++ix) {
                float radar_pos = -0.6f + ix * 1.2f / (Na-1);
                float R_radar = std::sqrt(std::pow(pixel_x, 2) + std::pow(pixel_y - radar_pos, 2));
                
                int N = static_cast<int>(std::abs(std::round(2 * R_radar * Kr / (c * fs / (up_n * Nr)))));
                
                if (N >= 0 && N < Nr * up_n) {
                    std::complex<float> sig = dechirp_data[ix * Nr * up_n + N];
                    float phase = -4.0f * M_PI * R_radar / lambda;
                    pixel_value += sig * std::exp(std::complex<float>(0.0f, phase));
                }
            }
            
            img[y_idx * N_x + x_idx] = std::abs(pixel_value);
        }
    }

    return img;
}

// BP���� GPU�汾
std::vector<float> bp_imaging_gpu(const std::vector<std::complex<float>>& dechirp_data) {
    // ����GPU�ڴ�
    float *d_img_real, *d_img_imag;
    float *d_dechirp_real, *d_dechirp_imag;
    float *d_y_radar, *d_X, *d_Y;
    
    cudaMalloc(&d_img_real, N_y * N_x * sizeof(float));
    cudaMalloc(&d_img_imag, N_y * N_x * sizeof(float));
    cudaMalloc(&d_dechirp_real, Na * Nr * 10 * sizeof(float));
    cudaMalloc(&d_dechirp_imag, Na * Nr * 10 * sizeof(float));
    cudaMalloc(&d_y_radar, Na * sizeof(float));
    cudaMalloc(&d_X, N_y * N_x * sizeof(float));
    cudaMalloc(&d_Y, N_y * N_x * sizeof(float));

    // ׼����������
    std::vector<float> y_radar(Na);
    std::vector<float> X(N_y * N_x), Y(N_y * N_x);
    for (int i = 0; i < Na; ++i) {
        y_radar[i] = -0.6f + i * 1.2f / (Na-1);
    }
    
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            X[i * N_x + j] = x_picture_min + j * dt_x;
            Y[i * N_x + j] = y_picture_min + i * dt_y;
        }
    }

    // �������ݵ�GPU
    std::vector<float> dechirp_real(Na * Nr * 10);
    std::vector<float> dechirp_imag(Na * Nr * 10);
    for (size_t i = 0; i < dechirp_data.size(); ++i) {
        dechirp_real[i] = dechirp_data[i].real();
        dechirp_imag[i] = dechirp_data[i].imag();
    }

    cudaMemcpy(d_dechirp_real, dechirp_real.data(), Na * Nr * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dechirp_imag, dechirp_imag.data(), Na * Nr * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_radar, y_radar.data(), Na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X.data(), N_y * N_x * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(), N_y * N_x * sizeof(float), cudaMemcpyHostToDevice);

    // ����CUDA����Ϳ��С
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N_y + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // ����kernel
    bpImagingKernel<<<numBlocks, threadsPerBlock>>>(
        d_img_real, d_img_imag, d_dechirp_real, d_dechirp_imag,
        d_y_radar, d_X, d_Y, 10, lambda, fs, Kr, c, Na, N_x, N_y
    );

    // ���kernelִ�д���
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // ��ȡ���
    std::vector<float> img_real(N_y * N_x);
    std::vector<float> img_imag(N_y * N_x);
    cudaMemcpy(img_real.data(), d_img_real, N_y * N_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(img_imag.data(), d_img_imag, N_y * N_x * sizeof(float), cudaMemcpyDeviceToHost);

    // �ͷ�GPU�ڴ�
    cudaFree(d_img_real);
    cudaFree(d_img_imag);
    cudaFree(d_dechirp_real);
    cudaFree(d_dechirp_imag);
    cudaFree(d_y_radar);
    cudaFree(d_X);
    cudaFree(d_Y);

    // �������
    std::vector<float> result(N_y * N_x);
    for (int i = 0; i < N_y * N_x; ++i) {
        result[i] = std::sqrt(img_real[i] * img_real[i] + img_imag[i] * img_imag[i]);
    }

    return result;
}

int main() {
    // ���ɻز�����
    std::cout << "Generating radar echo data..." << std::endl;
    auto dechirp_data = generate_echo_data();

    // CPU����
    std::cout << "Starting CPU computation..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto img_cpu = bp_imaging_cpu(dechirp_data);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count() / 1000.0;

    // GPU����
    std::cout << "Starting GPU computation..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto img_gpu = bp_imaging_gpu(dechirp_data);
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count() / 1000.0;

    // ��ӡ���ܱȽ�
    std::cout << "CPU computation time: " << time_cpu << " seconds" << std::endl;
    std::cout << "GPU computation time: " << time_gpu << " seconds" << std::endl;
    std::cout << "Speedup: " << time_cpu / time_gpu << "x" << std::endl;

    // ʹ��matplotlib-cpp��ͼ
    plt::figure_size(1200, 500);
    
    // ������������
    std::vector<std::vector<double>> img_cpu_2d(N_y, std::vector<double>(N_x));
    std::vector<std::vector<double>> img_gpu_2d(N_y, std::vector<double>(N_x));
    
    // �ҵ����ֵ
    double max_val_cpu = 0.0;
    double max_val_gpu = 0.0;
    for (size_t i = 0; i < img_cpu.size(); ++i) {
        max_val_cpu = std::max(max_val_cpu, std::abs(img_cpu[i]));
        max_val_gpu = std::max(max_val_gpu, std::abs(img_gpu[i]));
    }

    // ��һά����ת��Ϊ��ά����
    for (int i = 0; i < N_y; ++i) {
        for (int j = 0; j < N_x; ++j) {
            img_cpu_2d[i][j] = 20.0 * std::log10(std::abs(img_cpu[i * N_x + j]) / max_val_cpu);
            img_gpu_2d[i][j] = 20.0 * std::log10(std::abs(img_gpu[i * N_x + j]) / max_val_gpu);
        }
    }

    // ��������������
    std::vector<double> x(N_x), y(N_y);
    for (int i = 0; i < N_x; ++i) x[i] = x_picture_min + i * dt_x;
    for (int i = 0; i < N_y; ++i) y[i] = y_picture_min + i * dt_y;

    // ����CPU���
    plt::subplot(1, 2, 1);
    plt::imshow(img_cpu_2d, {
        {"cmap", "jet"},
        {"extent", std::vector<double>{x_picture_min, x_picture_max, y_picture_min, y_picture_max}},
        {"aspect", "auto"},
        {"vmin", -40.0}
    });
    plt::colorbar();
    plt::title("CPU BP Imaging Result");
    plt::xlabel("Range (m)");
    plt::ylabel("Azimuth (m)");

    // ����GPU���
    plt::subplot(1, 2, 2);
    plt::imshow(img_gpu_2d, {
        {"cmap", "jet"},
        {"extent", std::vector<double>{x_picture_min, x_picture_max, y_picture_min, y_picture_max}},
        {"aspect", "auto"},
        {"vmin", -40.0}
    });
    plt::colorbar();
    plt::title("GPU BP Imaging Result");
    plt::xlabel("Range (m)");
    plt::ylabel("Azimuth (m)");

    plt::tight_layout();
    plt::show();

    return 0;
} 