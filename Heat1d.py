import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return np.exp(- alpha * np.pi**2 * x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])

def diff_equation(x,y):
    dydt = dde.grad.jacobian(y,x,i=0,j=1)
    dydxx = dde.grad.hessian(y, x, i=0, j=0)
    return dydt - alpha * dydxx

def is_on_boundary(x, on_boundary):
    if not on_boundary:
        return False
    return np.isclose(x[0], 0) or np.isclose(x[0], 1)

def boundary_condition_value(x):
    return 0

def is_on_initial_condition(x, on_initial):
    return on_initial and dde.utils.isclose(x[1],0)

def initial_condition_value(x):
    return np.sin(np.pi * x[:, 0:1])

alpha = 0.4

geom = dde.geometry.Interval(0,1)
timedomain = dde.geometry.TimeDomain(0, 1) 
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, boundary_condition_value, is_on_boundary)
ic = dde.icbc.IC(geomtime, initial_condition_value, is_on_initial_condition)

data = dde.data.TimePDE(geomtime, diff_equation, [bc, ic], num_domain=4000, num_boundary=2000, num_initial=1000, solution=func, num_test=1000)

layer_size = [2] + [20] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)

model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=20000)

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
xx, tt = np.meshgrid(x, t)
X = np.vstack((xx.ravel(), tt.ravel())).T

y_pred = model.predict(X).reshape(100, 100)
y_exact = func(X).reshape(100, 100)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.contourf(xx, tt, y_exact, levels=50, cmap="jet")
plt.colorbar(label="u")
plt.title("Egzaktno rješenje")
plt.xlabel("x")
plt.ylabel("t")


plt.subplot(1, 2, 2)
plt.contourf(xx, tt, y_pred, levels=50, cmap="jet")
plt.colorbar(label="u")
plt.title("PINNs rješenje")
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()

time_slices = [0.0, 0.25, 0.50, 0.75, 1.0]
plt.figure(figsize=(12, 2 * len(time_slices)))  

for i, t_val in enumerate(time_slices):
    idx = np.argmin(np.abs(t - t_val))  
    y_exact_slice = y_exact[idx, :]
    y_pred_slice = y_pred[idx, :]

    plt.subplot(len(time_slices), 1, i + 1)
    plt.plot(x, y_exact_slice, 'r-', label="Egzaktno rješenje")
    plt.plot(x, y_pred_slice, 'b--', label="PINNs rješenje")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Rješenje za t = {t_val}")
    plt.legend()

plt.tight_layout()
plt.show()
