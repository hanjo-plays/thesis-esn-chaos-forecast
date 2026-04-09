# Conceptual sketch: two nearby trajectories diverging on the attractor
import matplotlib.pyplot as plt
import numpy as np

def bezier_curve(P0, P1, P2, n_points=100):
    t = np.linspace(0, 1, n_points)
    x = (1-t)**2 * P0[0] + 2*(1-t)*t * P1[0] + t**2 * P2[0]
    y = (1-t)**2 * P0[1] + 2*(1-t)*t * P1[1] + t**2 * P2[1]
    return x, y

# control points picked by hand to get the right visual
P0 = np.array([1.0, 2.0])
P2 = np.array([6.0, 4.0])
P1 = np.array([3.5, 3.0])

Q0 = np.array([0.8, 2.8])
Q2 = np.array([6.0, 7.5])
Q1 = np.array([3.0, 6.5])

x_lower, y_lower = bezier_curve(P0, P1, P2)
x_upper, y_upper = bezier_curve(Q0, Q1, Q2)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x_lower, y_lower, 'k-', linewidth=1.5)
ax.plot(x_upper, y_upper, 'k-', linewidth=1.5)

ax.plot(P0[0], P0[1], 'ko', markersize=6)
ax.plot(Q0[0], Q0[1], 'ko', markersize=6)
ax.plot(P2[0], P2[1], 'ko', markersize=6)
ax.plot(Q2[0], Q2[1], 'ko', markersize=6)

# Separation lines (epsilon) at start and end
ax.plot([P0[0], Q0[0]], [P0[1], Q0[1]], 'k-', linewidth=1)
ax.plot([P2[0], Q2[0]], [P2[1], Q2[1]], 'k-', linewidth=1)

# Direction arrows
dx_lower = x_lower[-1] - x_lower[-5]
dy_lower = y_lower[-1] - y_lower[-5]
dx_upper = x_upper[-1] - x_upper[-5]
dy_upper = y_upper[-1] - y_upper[-5]

ax.arrow(P2[0], P2[1], dx_lower, dy_lower, head_width=0.2, head_length=0.3, fc='k', ec='k')
ax.arrow(Q2[0], Q2[1], dx_upper, dy_upper, head_width=0.2, head_length=0.3, fc='k', ec='k')

ax.text(P0[0]+0.1, P0[1]-0.3, r'$x_0$', fontsize=14, ha='left', va='top')
ax.text(Q0[0]-0.2, Q0[1]+0.1, r'$\tilde{x}_0$', fontsize=14, ha='right', va='bottom')
ax.text(P2[0]+0.1, P2[1]-0.4, r'$x(t)$', fontsize=14, ha='left', va='top')
ax.text(Q2[0]+0.1, Q2[1]+0.2, r'$\tilde{x}(t)$', fontsize=14, ha='left', va='bottom')

mid_eps0 = (P0 + Q0) / 2
ax.text(mid_eps0[0]+0.2, mid_eps0[1], r'$\varepsilon_0$', fontsize=14, va='center', ha='left')

mid_epst = (P2 + Q2) / 2
ax.text(mid_epst[0]+0.2, mid_epst[1], r'$\varepsilon(t)$', fontsize=14, va='center', ha='left')

ax.set_xlim(-0.5, 8)
ax.set_ylim(0, 9)
ax.axis('off')

fig.text(0.05, 0.9, 'Fig. 1  Evolution of two close points on the attractor\ndynamic system', fontsize=12, fontweight='bold', ha='left', va='top')

plt.show()
