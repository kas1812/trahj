import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
def potential_function(k):
    x, y = sp.symbols('x y') 
    r = sp.sqrt(x**2 + y**2)
    V = -k / r 
    F_x = -sp.diff(V, x)
    F_y = -sp.diff(V, y)
    print("force_x:", F_x)
    print("force_y:", F_y)
    potential_eval = sp.lambdify((x, y), V, 'numpy')
    force_eval_x = sp.lambdify((x, y), F_x, 'numpy')
    force_eval_y = sp.lambdify((x, y), F_y, 'numpy')
    
    return potential_eval, force_eval_x, force_eval_y

vsx = []
vsy =[]
E = [0]
dt = 0.01
def generate_data(k=3, x=1.0, y=0.0, vx=0.0, vy=1.7, dt=0.01):
    rs = [np.sqrt(x**2 + y**2)]
    x_values = [x]
    y_values = [y]
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x_values, y_values, rs, vsx, vsy, E = generate_data()
time_values = np.arange(len(rs)) * dt

axes[0].plot(x_values, y_values, color='tab:blue', linewidth=2)
axes[0].set_xlabel('x position (m)')
axes[0].set_ylabel('y position (m)')
axes[0].set_title('Trajectory of a Particle in a Gravitational Potential')
axes[0].set_aspect('equal', adjustable='box')
axes[0].grid(True)

axes[1].plot(time_values, rs, color='tab:orange', linewidth=2)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('radius r (m)')
axes[1].set_title('Radial Distance vs Time')
axes[1].grid(True)

Energy_steps = np.arange(len(E))
plt.figure(figsize=(6, 4))  
plt.plot(rs, E, color='tab:purple', linewidth=2)
plt.xlabel('r')
plt.ylabel('energy E')
plt.title('Total Energy vs Time')
plt.grid(True)
"""acc_steps = np.arange(len(acx))
axes[2].plot(acc_steps, acx, color='tab:red', linewidth=2)
axes[2].set_xlabel('time step')
axes[2].set_ylabel('acceleration a_x')
axes[2].set_title('X-component of Acceleration')
axes[2].grid(True)

axes[3].plot(acc_steps, acy, color='tab:green', linewidth=2)
axes[3].set_xlabel('time step')
plt.tight_layout()"""
plt.show()