import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(t):
    A = 1.000
    k = 0.055
    w = 2.0
    return A * math.exp(k*t) * math.cos(w*t)

def midpoint_integration(a, b, n):
    h = (b-a)/n
    integral = 0
    for i in range(n):
        xi = a + (i+0.5)*h
        integral += f(xi)
    integral *= h
    return integral

def trapezoid_integration(a, b, n):
    h = (b-a)/n
    integral = (f(a) + f(b))/2
    for i in range(1, n):
        xi = a + i*h
        integral += f(xi)
    integral *= h
    return integral

def simpson_integration(a, b, n):
    h = (b-a)/n
    integral = f(a) + f(b)
    for i in range(1, n):
        xi = a + i*h
        if i%2 == 0:
            integral += 2*f(xi)
        else:
            integral += 4*f(xi)
    integral *= h/3
    return integral


def ground_truth_integration(A, k, t, w):
    numerator = A * math.exp(k*t) * (w*math.sin(w*t) + k*math.cos(w*t))
    denominator = w**2 + k**2
    result = numerator / denominator
    return result


def ground_truth_integration(a, b):
    integral = quad(f, a, b)
    return integral

Ta = 0.001
Tb = 2.0

midpoint_diffs = []
trapezoid_diffs = []
simpson_diffs = []

for k in range(2, 11, 2):
    n = 2**k
    print(f"\nFor N = {n}:")
    midpoint = midpoint_integration(Ta, Tb, n)
    trapezoid = trapezoid_integration(Ta, Tb, n)
    simpson = simpson_integration(Ta, Tb, n)
    ground_truth = ground_truth_integration(Ta, Tb)

    midpoint_diff = abs(midpoint - ground_truth[0])
    trapezoid_diff = abs(trapezoid - ground_truth[0])
    simpson_diff = abs(simpson - ground_truth[0])

    print(f"Midpoint integral = {midpoint:.6f}, difference = {(midpoint_diff):.6f}")
    print(f"Trapezoid integral = {trapezoid:.6f}, difference = {(trapezoid_diff):.6f}")
    print(f"Simpson integral = {simpson:.6f}, difference = {(simpson_diff):.6f}")
    print(f"Ground truth integral = {ground_truth[0]:.6f}")
    midpoint_diffs.append(midpoint_diff)
    trapezoid_diffs.append(trapezoid_diff)
    simpson_diffs.append(simpson_diff)

n_values = [2**k for k in range(2, 11, 2)]
plt.plot(n_values, midpoint_diffs, label='Midpoint', color = 'black')
plt.plot(n_values, trapezoid_diffs, label='Trapezoid', color = 'red')
plt.plot(n_values, simpson_diffs, label='Simpson', color = 'green')
plt.xscale('log', base=2)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Absolute difference from ground truth')
plt.title('Convergence of integration algorithms')
plt.legend()
plt.show()

