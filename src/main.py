import numpy as np
from numba import jit
import time


@jit
def calc_H(H, psi, time, k_range, delta_x, kappa, x_k, omega):
    H_temp = H
    for k in k_range:
        if k < 1 or k >= max(k_range):
            continue
        H_temp[k] = -0.5 / (delta_x ** 2) * (psi[k + 1] + psi[k - 1] - 2 * psi[k]) + \
                    + kappa * (x_k[k] - 0.5) * psi[k] * np.sin(omega * time)
    return H_temp


@jit
def main():
    # =======================
    N = 100
    n = 1
    kappa = 0
    omega = 0
    tau = 0
    delta_tau = 0.001
    # =======================

    delta_x = 1 / N
    k_range = np.arange(0, 101)
    x_k = np.linspace(0, 1, N + 1)

    psi_real = np.sqrt(2) * np.sin(n * np.pi * x_k)
    psi_imag = np.zeros(shape=x_k.shape)

    H_real = np.zeros(shape=x_k.shape)
    H_imag = np.zeros(shape=x_k.shape)

    NUM_STEPS = 1000
    for step in range(NUM_STEPS):
        tau += delta_tau
        psi_real += calc_H(H_imag, psi_imag, tau, k_range, delta_x, kappa, x_k, omega) * delta_tau / 2
        psi_imag -= calc_H(H_real, psi_real, tau + delta_tau / 2, k_range, delta_x, kappa, x_k, omega) * delta_tau
        psi_real += calc_H(H_imag, psi_imag, tau, k_range, delta_x, kappa, x_k, omega) * delta_tau / 2



if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))





    # for k in k_range:
    #     if k < 1 or k >= max(k_range):
    #         continue
    #     H_real[k] = -0.5 / (delta_x ** 2) * (psi_real[k + 1] + psi_real[k - 1] - 2 * psi_real[k]) + \
    #                 + kappa * (x_k - 0.5) * psi_real * np.sin(omega * tau)
    #     H_imag[k] = -0.5 / (delta_x ** 2) * (psi_imag[k + 1] + psi_imag[k - 1] - 2 * psi_imag[k]) + \
    #                 + kappa * (x_k - 0.5) * psi_imag * np.sin(omega * tau)

    # H_real = -0.5 / (delta_x ** 2) * np.array([psi_real[k + 1] + psi_real[k - 1] - 2 * psi_real[k]
    #                             if (0 < k < max(k_range))
    #                             else 0
    #                             for k in k_range]) + kappa * (x_k - 0.5) * psi_real * np.sin(omega * t)
    #
    # H_imag = -0.5 / (delta_x ** 2) * np.array([psi_imag[k + 1] + psi_imag[k - 1] - 2 * psi_imag[k]
    #                             if (0 < k < max(k_range))
    #                             else 0
    #                             for k in k_range]) + kappa * (x_k - 0.5) * psi_imag * np.sin(omega * t)
