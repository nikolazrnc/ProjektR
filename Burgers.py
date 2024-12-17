import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

def diff_equation(x, y):
    dydx = dde.grad.jacobian(y, x, i=0, j=0)
    dydt = dde.grad.jacobian(y, x, i=0, j=1)
    dydxx = dde.grad.hessian(y, x, i=0, j=0)
    return dydt + y * dydx - 0.01 / np.pi * dydxx

def is_on_boundary(x, on_boundary):
    if not on_boundary:
        return False
    return np.isclose(x[0], -1) or np.isclose(x[0], 1)

def boundary_condition_value(x):
    return 0

def is_on_initial_condition(x, on_initial):
    return on_initial and dde.utils.isclose(x[1], 0)

def initial_condition_value(x):
    return -np.sin(np.pi * x[:, 0:1])

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, boundary_condition_value, is_on_boundary)
ic = dde.icbc.IC(geomtime, initial_condition_value, is_on_initial_condition)

data = dde.data.TimePDE(geomtime, diff_equation, [bc, ic], num_domain=2600, num_boundary=100, num_initial=200)

layer_size = [2] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=20000)

t = np.linspace(0, 1, 100)
x = np.linspace(-1, 1, 300)
xx, tt = np.meshgrid(x, t)
X = np.vstack((xx.ravel(), tt.ravel())).T

y_pred = model.predict(X)

y_pred_reshaped = y_pred.reshape(len(t), len(x))

plt.figure(figsize=(12, 4))

plt.contourf(xx, tt, y_pred_reshaped, levels=50, cmap="jet")
plt.colorbar(label="u")
plt.title("PINNs rješenje")
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()

time_slices = [0.0, 0.25, 0.50, 0.75, 1.0]

num_plots = len(time_slices)
cols = 2  
rows = (num_plots + cols - 1) // cols  

plt.figure(figsize=(12, 3 * rows)) 

for i, t_val in enumerate(time_slices):
    idx = np.argmin(np.abs(t - t_val))  
    y_pinn = y_pred_reshaped[idx, :]  

    plt.subplot(rows, cols, i + 1)  
    plt.plot(x, y_pinn, 'b--', label="PINNs rješenje")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Rješenje za t = {t_val}")
    plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.4, top=0.92)

plt.show()

