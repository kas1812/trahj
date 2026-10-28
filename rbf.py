import numpy as np
import data_gen as dg
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
k =5
trasj_data = dg.traj_fam(k)
d_t =0.01
all_x, all_y, all_v = [], [], []
for x_values, y_values, rs, vsx, vsy, E, a_x, a_y in trasj_data:
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    vx_arr=np.array(vsx)
    vy_arr=np.array(vsy)
    #E_arr=np.array(E[1:])
    a_X = (x_arr[2:] - 2*x_arr[1:-1] + x_arr[:-2] )/ d_t**2
    a_Y = (y_arr[2:] - 2*y_arr[1:-1] + y_arr[:-2]) / d_t**2
    xs, ys = x_arr[1:-1], y_arr[1:-1]
    if np.hypot(xs, ys).min() < 0.088:
        continue
    dx, dy = np.diff(xs), np.diff(ys)
    dV0 = -(a_X[0] * (x_arr[1] - x_arr[0]) + a_Y[0] * (y_arr[1] - y_arr[0]))
    dV = -(a_X[:-1] + a_X[1:])*dx/2 - (a_Y[:-1] + a_Y[1:])*dy/2
    V_est = -k + dV0 + np.concatenate(([0], np.cumsum(dV)))
    #V_est = E_arr - 0.5 * (vx_arr**2 + vy_arr**2)

    all_x.append(x_arr[1:-1])
    all_y.append(y_arr[1:-1])
    all_v.append(V_est)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_v = np.concatenate(all_v)
print(f"Data points: {len(all_x)}")
print(f"Potential values: {len(all_v)}")
r_fit = np.sqrt(all_x**2 + all_y**2)
mask = (r_fit < 5.0)
if not np.any(mask):
    raise ValueError('The radius mask selected no training points.')

points = np.column_stack((all_x[mask], all_y[mask]))
rbf_fit = RBFInterpolator(points, all_v[mask], kernel='thin_plate_spline', neighbors=10, smoothing=0.0)

xg, yg = np.meshgrid(np.linspace(all_x[mask].min(), all_x[mask].max(), 100), np.linspace(all_y[mask].min(), all_y[mask].max(), 100))
query = np.column_stack([xg.ravel(), yg.ravel()])
v_pred = rbf_fit(query).reshape(xg.shape)
r_grid = np.sqrt(xg**2 + yg**2)
v_true = -k/r_grid
figg = plt.figure(figsize=(13,6))
ax1 = figg.add_subplot(1,2,1, projection ='3d')
ax1.plot_surface(xg, yg, v_pred, cmap='viridis', edgecolor='none', alpha=0.9)
ax2 = figg.add_subplot(1,2,2, projection='3d')
ax2.plot_surface(xg, yg, v_true, cmap='viridis', edgecolor='none')
ax1.set_title('RBF Interpolated Potential Function')
ax2.set_title('True Potential Function')
print( np.max(E) - np.min(E))
plt.tight_layout()
plt.show()