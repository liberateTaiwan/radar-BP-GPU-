#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <corecrt_math_defines.h>

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
const double theta = 20 * M_PI / 180;  // ������� 20�ȣ�ת��Ϊ���ȣ�
const int length_label = Nr;  

// ʹ�� using �򻯸������͵�д��
using Complex = std::complex<double>;

// ���ɻز�����
Eigen::MatrixXcd generate_echo_data(const std::vector<double>& t_label, 
                                  const std::vector<double>& y_radar,
                                  const std::pair<double, double>& target_pos) {
    std::vector<Complex> phase_carry(length_label);
    std::vector<Complex> sig_origin(length_label);
    
    // ���ɷ����ź�
    for (int i = 0; i < length_label; ++i) {
        double phase = 2 * M_PI * t_label[i] * fc;
        phase_carry[i] = Complex(cos(phase), sin(phase));
        sig_origin[i] = (std::abs(t_label[i]) <= PRT / 2) ? 
            Complex(cos(M_PI * Kr * t_label[i] * t_label[i]), 
                   sin(M_PI * Kr * t_label[i] * t_label[i])) * phase_carry[i] : 
            Complex(0, 0);
    }

    const int up_n = 10;
    Eigen::MatrixXcd dechirp_matrix(Na, length_label * up_n);
    dechirp_matrix.setZero();

    // ���ɻز�����
    for (int m = 0; m < Na; ++m) {
        double y_m = y_radar[m];
        double R = std::sqrt((y_m - target_pos.second) * (y_m - target_pos.second) + 
                            target_pos.first * target_pos.first);
        double delay = 2 * R / c;

        double phi = std::atan2(target_pos.second - y_m, target_pos.first);
        if (std::abs(phi) <= theta / 2) {
            std::vector<Complex> t_delay(length_label);
            std::vector<Complex> phase_carry_rx(length_label);
            std::vector<Complex> s_rx(length_label);
            
            for (int i = 0; i < length_label; ++i) {
                t_delay[i] = Complex(t_label[i] + delay, 0);
                phase_carry_rx[i] = Complex(cos(2 * M_PI * t_delay[i].real() * fc), 
                                          sin(2 * M_PI * t_delay[i].real() * fc));
                s_rx[i] = (std::abs(t_delay[i].real()) <= PRT / 2) ? 
                    Complex(cos(M_PI * Kr * t_delay[i].real() * t_delay[i].real()),
                           sin(M_PI * Kr * t_delay[i].real() * t_delay[i].real())) * 
                    phase_carry_rx[i] : Complex(0, 0);
            }

            // dechirp����
            std::vector<Complex> sig_rd(length_label);
            for (int i = 0; i < length_label; ++i) {
                sig_rd[i] = s_rx[i] * std::conj(sig_origin[i]);
            }

            // ����������
            std::vector<Complex> sig_rd_padded(length_label * up_n, Complex(0, 0));
            for (int i = 0; i < length_label / 2; ++i) {
                sig_rd_padded[i] = sig_rd[i];
            }
            for (int i = length_label / 2; i < length_label; ++i) {
                sig_rd_padded[i + length_label * up_n - length_label] = sig_rd[i];
            }

            // FFT����
            cufftHandle plan;
            cufftComplex* d_data;
            cudaMalloc((void**)&d_data, length_label * up_n * sizeof(cufftComplex));
            cudaMemcpy(d_data, sig_rd_padded.data(), 
                      length_label * up_n * sizeof(cufftComplex), 
                      cudaMemcpyHostToDevice);

            cufftPlan1d(&plan, length_label * up_n, CUFFT_C2C, 1);
            cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);

            std::vector<Complex> SIG_rd(length_label * up_n);
            cudaMemcpy(SIG_rd.data(), d_data, 
                      length_label * up_n * sizeof(cufftComplex), 
                      cudaMemcpyDeviceToHost);

            cufftDestroy(plan);
            cudaFree(d_data);

            for (int i = 0; i < length_label * up_n; ++i) {
                dechirp_matrix(m, i) = SIG_rd[i];
            }
        }
    }

    return dechirp_matrix;
}

// CUDA�˺�����BP����
__global__ void bp_imaging_kernel(Complex* img, const Complex* dechirp_data,
                                const double* y_radar, const double* X, const double* Y,
                                int N_y, int N_x, int Na, double lambda, double fs,
                                double Kr, double c, int up_n, int length_label) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx < N_x && y_idx < N_y) {
        Complex pixel_value(0, 0);
        double pixel_x = X[y_idx * N_x + x_idx];
        double pixel_y = Y[y_idx * N_x + x_idx];

        for (int ix = 0; ix < Na; ++ix) {
            double RadarPosNow = y_radar[ix];
            double R_radar = std::sqrt((pixel_x) * (pixel_x) + 
                                     (pixel_y - RadarPosNow) * (pixel_y - RadarPosNow));
            int N = static_cast<int>(std::abs(std::round(2 * R_radar * Kr / 
                                   (c * fs / (up_n * length_label)))));

            if (N >= 0 && N < length_label * up_n) {
                Complex sig = dechirp_data[ix * length_label * up_n + N];
                double phase = -4 * M_PI * R_radar / lambda;
                pixel_value += sig * Complex(cos(phase), sin(phase));
            }
        }

        img[y_idx * N_x + x_idx] = pixel_value;
    }
}

int main() {
    // ��ʼ��ʱ����
    std::vector<double> t_label(length_label);
    for (int i = 0; i < length_label; ++i) {
        t_label[i] = -PRT / 2 + i / fs;
    }

    // ��ʼ���״�λ��
    std::vector<double> y_radar(Na);
    for (int i = 0; i < Na; ++i) {
        y_radar[i] = -0.6 + (0.6 - (-0.6)) * i / (Na - 1);
    }

    // ���ɻز�����
    std::cout << "Generating radar echo data..." << std::endl;
    auto dechirp_data = generate_echo_data(t_label, y_radar, {500, 0});

    // CPU����ʱ��
    auto start_cpu = std::chrono::high_resolution_clock::now();
    // ... CPU������� ...
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // GPU����ʱ��
    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);
    cudaEventRecord(start_gpu);
    // ... GPU������� ...
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, end_gpu);
    double gpu_time = gpu_time_ms / 1000.0;

    // ��ӡ���ܱȽ�
    std::cout << "\nCPU computation time: " << cpu_time.count() << " seconds" << std::endl;
    std::cout << "GPU computation time: " << gpu_time << " seconds" << std::endl;
    std::cout << "Speedup: " << cpu_time.count() / gpu_time << "x" << std::endl;

    return 0;
}