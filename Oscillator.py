import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

m = 1
mu = 3
k = 1

x0, v0 = 0, 2

delta = mu / (2*m)
w0 = np.sqrt(k/m)

def diff_equation(t,x):
        dxdt = dde.grad.jacobian(x,t)
        dxdtt = dde.grad.hessian(x,t)
        return m * dxdtt + mu * dxdt + k*x

def func(t):
    delta = mu / (2 * m)
    w0 = np.sqrt(k / m)
    if delta < w0:  # slabo prigušene oscilacije
        wd = np.sqrt(w0**2 - delta**2)
        return np.exp(-delta * t) * (x0 * np.cos(wd * t) + (v0 + delta * x0) / wd * np.sin(wd * t))
    elif delta == w0: # kritično prigušenje
        return (x0 + (v0 + delta * x0) * t) * np.exp(-delta * t)
    else: # jako prigušene oscilacije
        r1 = -delta + np.sqrt(delta**2 - w0**2)
        r2 = -delta - np.sqrt(delta**2 - w0**2)
        C1 = (v0 + delta * x0 - r2 * x0) / (r1 - r2)
        C2 = (-v0 - delta * x0 + r1 * x0) / (r1 - r2)
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

def boundary_left(t, on_initial):
    return on_initial and np.isclose(t[0], 0)

def bc_func1(inputs, outputs, X):
    return outputs - x0

def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0,j=None) - v0

interval = dde.geometry.TimeDomain(0, 10)

ic1 = dde.icbc.OperatorBC(interval, bc_func1, boundary_left)
ic2 = dde.icbc.OperatorBC(interval, bc_func2, boundary_left)

data = dde.data.TimePDE(interval, diff_equation, [ic1, ic2], 100, 20, solution=func, num_test=100)

layers = [1] + [30] * 2 + [1]
activation = "tanh"
init = "Glorot uniform"
net = dde.nn.FNN(layers, activation, init)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=30000)

t = np.linspace(0, 10, 100).reshape(100,1)
x_pred = model.predict(t)

x_true = func(t)

plt.plot(t, x_pred, label='PINNs rješenje', linestyle='-', color='blue', linewidth=2)
plt.plot(t, x_true, label='Egzaktno rješenje', linestyle='--', color='red', linewidth=2)

plt.xlabel('Vrijeme t', fontsize=12)
plt.ylabel('Pomak x(t)', fontsize=12)
plt.title('Usporedba predviđenog i egzaktnog rješenja', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

plt.gca().set_facecolor('lightgray')
plt.grid(True, which='both', linestyle=':', color='black')
plt.tick_params(axis='both', which='major', labelsize=10)

plt.show()