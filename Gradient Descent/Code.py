import numpy as np
import matplotlib.pyplot as plt

# Define the cost/loss function and its derivative
def f(Wstart):
    return Wstart[0] ** 2 + Wstart[0] * Wstart[1] + 2 * Wstart[1] ** 2
def df(Wstart):
    return np.array([2 * Wstart[0] + Wstart[1], Wstart[0] + 4 * Wstart[1]])
# Set the hyperparameters
learning_rates = [0.01, 0.1, 0.5]
Nmax = 1000
toler = 1e-6
for learning_rate in learning_rates:
    # Set the initial point and initialize variables
    Wstart = np.array([1.0, 1.0])
    f_values = [f(Wstart)]
    step = 0
    # Run gradient descent
    while step < Nmax:
        # Compute the gradient and update Wstart
        grad = df(Wstart)
        Wstart_new = Wstart - learning_rate * grad

        # Compute the new function value and check for convergence
        f_new = f(Wstart_new)
        if abs(f_new - f_values[-1]) < toler:
            break
        # Update variables for next iteration
        Wstart = Wstart_new
        f_values.append(f_new)
        step += 1

    # Print the results
    print(f"Lambda: {learning_rate}")
    print(f"Optimized solution: w1 = {Wstart[0]}, w2 = {Wstart[1]}, Fmin(w1,w2) = {f_values[-1]}")
    print(f"Number of steps: {step}")

    # Extract w1 and w2 values from f_values list
    w1 = f_values[:-1]
    w2 = f_values[1:]

    # Plot w1 and w2 values
    # Plot trajectory
    plt.figure()
    plt.plot(w1, w2, 'bo-')
    plt.plot(w1, w2, 'r', label='Trajectory')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title(f'Gradient Descent Trajectory (lambda={learning_rate})')
    plt.legend()
    plt.show()
