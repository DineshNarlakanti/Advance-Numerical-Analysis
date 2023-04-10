import numpy as np
import matplotlib.pyplot as plt

# Define the given parameters
A = 1.0
k = 0.055
w = 2.0
Ta = 0.001
Tb = 2.0

# Define the function
def f(t):
    return A * np.exp(k*t) * np.cos(w*t)

# Define the first and second derivatives as ground truths
def dfdt(t):
    return -A * np.exp(t) * (w*t*np.sin(w*t) - np.cos(w*t))

def d2fdt2(t):
    return -A*np.exp(k) * ((w*t)*np.sin(w*t) - np.cos(w*t))

# Define range of N values to test
N_range = [2**n for n in range(2, 11,2)]

# Calculate ground truth values
t = 1.0
gt_dFdt = dfdt(t)
gt_d2Fdt2 = d2fdt2(t)

# Initialize arrays to store errors
cd_dFdt_error = np.zeros(len(N_range))
cd_d2Fdt2_error = np.zeros(len(N_range))
fd_dFdt_error = np.zeros(len(N_range))
fd_d2Fdt2_error = np.zeros(len(N_range))

# Calculate errors for each N value using central difference method
for i, N in enumerate(N_range):
    h = t/N
    cd_dFdt_approx = (f(t+h) - f(t-h))/(2*h)
    cd_d2Fdt2_approx = (f(t+h) - 2*f(t) + f(t-h))/(h**2)
    cd_dFdt_error[i] = abs(cd_dFdt_approx - gt_dFdt)
    cd_d2Fdt2_error[i] = abs(cd_d2Fdt2_approx - gt_d2Fdt2)




# Calculate errors for each N value using forward difference method
for i, N in enumerate(N_range):
    h = t/N
    fd_dFdt_approx = (f(t+h) - f(t))/h
    fd_d2Fdt2_approx = (f(t+2*h) - 2*f(t+h) + f(t))/(h**2)
    fd_dFdt_error[i] = abs(fd_dFdt_approx - gt_dFdt)
    fd_d2Fdt2_error[i] = abs(fd_d2Fdt2_approx - gt_d2Fdt2)
    print(f"\nFor N = {N}:")
    # Print the values of ground truth, forward difference, central difference, and error
    print("dF/dt:")
    print(f"Forward Difference = {fd_dFdt_approx}, difference = {max(fd_dFdt_error)}")
    print(f"Central Difference = {cd_dFdt_approx}, difference = {max(cd_dFdt_error)}")
    print(f"Ground truth integral = {gt_dFdt}")

    print("d2F/dt2:")
    print(f"Forward Difference = {fd_d2Fdt2_approx}, difference = {max(fd_d2Fdt2_error)}")
    print(f"Central Difference = {cd_d2Fdt2_approx}, difference = {max(cd_d2Fdt2_error)}")
    print(f"Ground truth integral = {gt_d2Fdt2}")






# Plot errors
plt.plot(N_range, cd_dFdt_error, label='Central diff')
plt.plot(N_range, fd_dFdt_error,  label='Forward diff')
plt.xscale('log', base=2)

plt.title('Absolute Error in Approximating First Derivative')
plt.xlabel('N')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()

plt.plot(N_range, cd_d2Fdt2_error, label='Central diff')
plt.plot(N_range, fd_d2Fdt2_error, label='Forward diff')
plt.xscale('log', base=2)

plt.title('Absolute Error in Approximating Second Derivative')
plt.xlabel('N')
plt.ylabel('Absolute Error')
plt.legend()
plt.show()
