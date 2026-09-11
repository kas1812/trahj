import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
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
def generate_data(k=3, x=1.0, y=0.0, vx=0.0, vy=1.7, dt=0.01):
    rs = [np.sqrt(x**2 + y**2)]
    x_values = [x]
    y_values = [y]
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

def traj_fam():
    num_targets = 100
    trajs = []
    for i in range(num_targets):
        E_target = np.random.uniform(-3, -1)
        vy = np.sqrt(2*(E_target + 3/np.sqrt(1**2 + 0.0**2)))
        x_values, y_values, rs, vsx, vsy, E = generate_data(k=3, x=1.0, y=0.0, vx=0.0, vy=vy, dt=0.01)
        trajs.append((x_values, y_values, rs, vsx, vsy, E))
    return trajs

