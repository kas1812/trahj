import numpy as np
import data_gen as dg
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
trasj_data = dg.traj_fam()

all_x, all_y, all_v = [], [], []
for x_values, y_values, rs, vsx, vsy, E in trasj_data:
    x_arr = np.array(x_values)
    y_arr = np.array(y_values)
    vx_arr=np.array(vsx)
    vy_arr=np.array(vsy)
    E_arr=np.array(E[1:])
    V_est = E_arr - 0.5 * (vx_arr**2 + vy_arr**2)
    all_x.append(x_arr)
    all_y.append(y_arr)
    all_v.append(V_est)

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)
all_v = np.concatenate(all_v)
print(f"Data points: {len(all_x)}")
print(f"Potential values: {len(all_v)}")
r_fit = np.sqrt(all_x**2 + all_y**2)
mask = (r_fit > 0.1) & (r_fit < 5.0)
if not np.any(mask):
    raise ValueError('The radius mask selected no training points.')

points = np.column_stack((all_x[mask], all_y[mask]))
rbf_fit = RBFInterpolator(points, all_v[mask], kernel='thin_plate_spline', neighbors=10, smoothing=0.0)

xg, yg = np.meshgrid(np.linspace(all_x.min(), all_x.max(), 100), np.linspace(all_y.min(), all_y.max(), 100))
query = np.column_stack([xg.ravel(), yg.ravel()])
v_pred = rbf_fit(query).reshape(xg.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xg, yg, v_pred, levels=50, cmap='viridis')
plt.colorbar(label='Potential V')
plt.scatter(all_x[mask], all_y[mask], c='red', s=1, alpha=0.5)
plt.title('RBF Interpolated Potential Function')
plt.gca()
plt.show()