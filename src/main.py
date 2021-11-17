import numpy as np
from numba import jit
import time
from tqdm import tqdm
import os
import pandas as pd

# =======================
N = 100         # number of ranges
n = 1           # which energy level
kappa = 0
omega = 0 # 3 * np.pi**2 / 2
tau = 0
delta_tau = 0.0001
# =======================

delta_x = 1 / N
k_range = np.arange(0, 101)
x_k = np.linspace(0, 1, N + 1)

params = [k_range, delta_x, kappa, x_k, omega, delta_tau]

psi_real = np.sqrt(2) * np.sin(n * np.pi * x_k)
psi_imag = np.zeros(shape=x_k.shape)

H_real = np.zeros(shape=x_k.shape)
H_imag = np.zeros(shape=x_k.shape)


@jit
def calc_H(psi, time):
    H = np.zeros(shape=x_k.shape)
    for k in range(N+1):
        if k < 1 or k >= N:
            continue
        H[k] = -0.5 / (delta_x ** 2) * (psi[k + 1] + psi[k - 1] - 2 * psi[k]) + \
                    + kappa * (x_k[k] - 0.5) * psi[k] * np.sin(omega * time)
    H[0] = 0
    H[-1] = 0
    return H


@jit
def make_calculations(psi_real, psi_imag, tau):
    # psi_real(tau + delta_tau/2)
    psi_real += calc_H(psi_imag, tau) * delta_tau / 2

    H_real_plus_half_tau = calc_H(psi_real, tau + delta_tau / 2)
    # psi_imag(tau + delta_tau)
    psi_imag -= H_real_plus_half_tau * delta_tau
    
    H_imag_plus_tau = calc_H(psi_imag, tau + delta_tau)
    # psi_real(tau + delta_tau)
    psi_real += H_imag_plus_tau * delta_tau / 2
    
    return psi_real, psi_imag


# def write_rho_to_file(rho):
#     with open(os.path.join("..", "data", "rho.txt"), 'a') as f:
#         pd.DataFrame(rho).transpose(copy=True).to_csv(f, sep=';', index=False, header=None, line_terminator='\n')
def write_rho_to_file(rho):
    with open(os.path.join("..", "data", "rho.txt"), 'a') as f:
        pd.DataFrame(rho).transpose(copy=True).to_csv(f, sep=';', index=False, header=None, line_terminator='\n')


def write_params_to_file(norm, x, epsilon, write_headers=False):
    with open(os.path.join("..", "data", "temp_values.txt"), 'a') as f:
        if write_headers:
            f.write("norm;x;epsilon\n")
        f.write("{};{};{}\n".format(norm, x, epsilon))



def main():
    tau = 0

    psi_real = np.sqrt(2) * np.sin(n * np.pi * x_k)
    psi_imag = np.zeros(shape=x_k.shape)

    H_real = np.zeros(shape=x_k.shape)
    H_imag = np.zeros(shape=x_k.shape)

    rho_arr = np.zeros(shape=(2,x_k.shape[0]))

    write_headers = True
    NUM_STEPS = 4000
    for step in tqdm(range(NUM_STEPS)):
        H_real = calc_H(psi_real, tau)
        H_imag = calc_H(psi_imag, tau)

        # if step % 2 == 0:
        #     norm = delta_x * np.sum(psi_real**2 + psi_imag**2)
        #     x = delta_x * np.sum(x_k * (psi_real**2 + psi_imag**2))
        #     epsilon = delta_x * np.sum(psi_real * H_real + psi_imag * H_imag)

        psi_real, psi_imag = make_calculations(psi_real, psi_imag, tau)

        rho = psi_real**2 + psi_imag**2

        if step % 10 == 0:
            rho_arr = np.vstack([rho_arr, rho])
            # write_rho_to_file(rho)
            # write_params_to_file(norm, x, epsilon, write_headers=write_headers)
        
        write_headers = False
        tau += delta_tau

    write_rho_to_file(rho_arr)



if __name__ == '__main__':
    start_time = time.time()

    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    rho_file = os.path.join("..", "data", "rho.txt")
    values_file = os.path.join("..", "data", "temp_values.txt")


    if os.path.exists(rho_file):
        os.remove(rho_file)
    if os.path.exists(values_file):
        os.remove(values_file)

    main()
    print("--- %s seconds ---" % (time.time() - start_time))

