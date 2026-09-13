import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
def potential_function(k):
    x, y = sp.symbols('x y') 
    r = sp.sqrt(x**2 + y**2)
    V = -k / r 
    F_x = -sp.diff(V, x)
    F_y = -sp.diff(V, y)
    potential_eval = sp.lambdify((x, y), V, 'numpy')
    force_eval_x = sp.lambdify((x, y), F_x, 'numpy')
    force_eval_y = sp.lambdify((x, y), F_y, 'numpy')
    return potential_eval, force_eval_x, force_eval_y
dt = 0.01

def generate_data(k, x, y, vx, vy, dt):
    rs = [np.sqrt(x**2 + y**2)]
    x_values = []
    y_values = []
    vsx = []
    vsy =[]
    E = [0]
    potential_eval, force_eval_x, force_eval_y = potential_function(k)

    a_x = force_eval_x(x, y)
    a_y = force_eval_y(x, y)
    for i in range(600):
        x += vx*dt + 0.5 * a_x * dt**2
        y += vy*dt + 0.5 * a_y * dt**2
        ax_new = force_eval_x(x, y)
        ay_new = force_eval_y(x, y)
        vx += 0.5 * (a_x + ax_new) * dt
        vy += 0.5 * (a_y + ay_new) * dt
        a_x, a_y = ax_new, ay_new
        r_new= np.sqrt(np.array(x)**2 + np.array(y)**2)
        E.append(0.5 * (vx**2 + vy**2) + potential_eval(x, y))
        x_values.append(x)
        y_values.append(y)
        rs.append(r_new)
        vsx.append(vx)
        vsy.append(vy)
    return x_values, y_values, rs, vsx, vsy, E

def traj_fam(k):
    num_targets = 2000
    trajs = []
    for i in range(num_targets):
        E_target = np.random.uniform(-7, 2)
        theta = np.random.uniform(0, 2*np.pi)
        speed = np.sqrt(2*(E_target + k/1))
        if speed ** 2 < 0.2:
            continue
        x0 = 1 * np.cos(theta)
        y0 = 1 * np.sin(theta)
        v_x = -speed *np.sin(theta)
        v_y = speed *np.cos(theta) 
        x_values, y_values, rs, vsx, vsy, E = generate_data(k, x0, y0, v_x, v_y, dt)
        trajs.append((x_values, y_values, rs, vsx, vsy, E))
    return trajs

frames = traj_fam(5)
for trajectory in frames:
    x_values, y_values, *_ = trajectory
    plt.plot(x_values, y_values)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()