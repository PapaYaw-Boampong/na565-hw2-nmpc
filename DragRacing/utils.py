import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time
import os
import sys
from contextlib import redirect_stdout, redirect_stderr, contextmanager

@contextmanager
def silence_output():
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull) as devnull, redirect_stderr(sys.stdout):
            yield

param = { "m":1778., "Izz":3049.,"L_f":1.094-0.1, "L_r":1.536+0.1, "hcg":0.55, "Peng": 172. * 1000, "Frr": 218., "Cd": 0.4243,
          "delta_max": 0.4712, "ddelta_max":0.3491, "C_alpha_f": 180. * 1000,"C_alpha_r": 400. * 1000,
         "mu_f": 0.75,"mu_r": 0.6, "g":9.81, "mu_lim": 0.8}

## distribution of traction force
def chi_fr(Fx):

    ## front-drive
    # xf = 0.125 * ca.tanh(2 * (Fx + 0.5)) + 0.875
    # xr = 1. - xf

    ## rear-drive
    xr = 0.125 * ca.tanh(2 * (Fx + 0.5)) + 0.875
    xf = 1. - xr

    return xf * Fx, xr * Fx

def get_slip_angle(Ux, Uy, r, delta, param):
    L_r = param["L_r"]
    L_f = param["L_f"]
    
    a_f = ca.arctan2((Uy + L_f * r), Ux) - delta
    a_r = ca.arctan2((Uy - L_r * r), Ux)
    return a_f, a_r

def tire_model_sim(alpha, Fz, Fx, C_alpha, mu):

    Fy_max_sq = (mu * Fz)**2 - (Fx)**2
    Fy_max = ca.if_else( Fy_max_sq <=0, 0, ca.sqrt(Fy_max_sq))
    
    alpha_slip = ca.arctan(3 * Fy_max / C_alpha)
    Fy = ca.if_else(ca.fabs(alpha) <= alpha_slip, 
        - C_alpha * ca.tan(alpha) 
        + C_alpha**2 * ca.fabs(ca.tan(alpha)) * ca.tan(alpha) / (3 * Fy_max)
        - C_alpha**3 * ca.tan(alpha)**3 / (27 * Fy_max**2), 
        - Fy_max * ca.sign(alpha))
    
    return Fy

def tire_model_ctrl(alpha, Fz, Fx, C_alpha, mu):
    # for each tire
    xi = 0.85

    ## NaN
    # Fy_max = ca.sqrt((mu * Fz)**2 - (0.99 * Fx)**2)
    # Fy_max = ca.sqrt((mu * Fz)**2 - (0.99 * Fx)**2)

    ## if else
    # Fy_max_sq = (mu * Fz)**2 - (0.99 * Fx)**2
    # Fy_max = ca.if_else( Fy_max_sq <=0, 0, ca.sqrt(Fy_max_sq))

    F_offset = 2000
    ## hyperbola
    Fy_max_sq = (mu * Fz)**2 - (0.99 * Fx)**2
    Fy_max_sq = (ca.sqrt( Fy_max_sq**2 + F_offset) + Fy_max_sq) / 2
    Fy_max = ca.sqrt(Fy_max_sq)

    alpha_mod = ca.arctan(3 * Fy_max / C_alpha * xi)
    
    Fy = ca.if_else(ca.fabs(alpha) <= alpha_mod, - C_alpha * ca.tan(alpha) 
        + C_alpha**2 * ca.fabs(ca.tan(alpha)) * ca.tan(alpha) / (3 * Fy_max)
        - C_alpha**3 * ca.tan(alpha)**3 / (27 * Fy_max**2), 
        - C_alpha * (1 - 2 * xi + xi**2) * ca.tan(alpha)
        - Fy_max * (3 * xi**2 - 2 * xi**3) * ca.sign(alpha))
    return Fy

def normal_load(Fx, param):
    # for both tires
    L_r = param["L_r"]
    L_f = param["L_f"]
    m = param["m"]
    g = param["g"]
    hcg = param["hcg"]
    
    L = (L_r + L_f)
    F_zf = L_r / L * m * g - hcg / L * Fx
    F_zr = L_f / L * m * g + hcg / L * Fx
    
    return F_zf, F_zr

def plot_results(T_sim, sim_time, x_log, u_log, tire_force_log):
    # trajectory in the world frame
    plt.figure(figsize=(20, 10))
    plt.plot(x_log[3, 0], x_log[4, 0], "r.", markersize = 20)
    plt.plot(x_log[3, :], x_log[4, :])
    plt.axis("scaled")

    plt.title("Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")

    theta = np.linspace(0, np.pi * 2, 100)
    plt.plot(np.sin(theta) * 10 + 500, np.cos(theta) * 10)
    plt.ylim([np.min(x_log[4, :]) -10, np.max(x_log[4, :]) + 10])

    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 3))
    plt.plot(sim_time, x_log[5])
    plt.xlim([0, T_sim])
    plt.title("Yaw Angle")
    plt.xlabel("Time(s)")
    plt.ylabel("rad")
    plt.grid()
    plt.show()

    # Velocity
    plt.figure(figsize=(20, 5))
    plt.plot(sim_time, x_log[0, :], label = "$U_x$" )
    plt.xlim([0, T_sim])

    plt.title("Longitudinal Velocity")
    plt.xlabel("Time(s)")
    plt.ylabel("m/s")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(sim_time, x_log[1, :], label = "$U_y$" )
    plt.plot(sim_time, x_log[2, :], label = "$\dot{r}$" )
    plt.xlim([0, T_sim])

    plt.title("Lateral Velocity / Yaw rate")
    plt.xlabel("Time(s)")
    plt.ylabel("m/s (rad/s)")
    plt.legend()
    plt.grid()
    plt.show()

    # Control input
    print("The traction force will decrease due to the engine power limits when the velocity is large.")
    rk_interval = 10
    plt.figure(figsize=(20, 5))
    plt.plot(sim_time[1::rk_interval-1], u_log[1, 1:])
    plt.xlim([0, T_sim])
    plt.title("Steering Angle")
    plt.xlabel("Time(s)")
    plt.ylabel("rad")
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(sim_time[1::rk_interval-1], u_log[0, 1:] / 1000)
    plt.xlim([0, T_sim])
    plt.title("Traction Force")
    plt.xlabel("Time(s)")
    plt.ylabel("kN")
    plt.grid()
    plt.show()

    # Tire force
    print("For successful trial, the tire forces should be distributed at the boundary of the friction cone to achieve maximum acceleration and braking. This behavior is similar to real-world drag racing.")
    rk_interval = 2
    plt.figure(figsize=(20, 4))
    plt.plot(sim_time[1::rk_interval-1], tire_force_log[0:3, 1:].T / 1000)
    plt.title("Front Tire Forces")
    plt.legend(["$F_x$", "$F_y$", "$F_z$"])
    plt.ylabel("Force (kN)")
    plt.xlabel("Time (s)")
    plt.xlim([0, T_sim])
    plt.grid()
    plt.show()

    plt.figure(figsize=(20, 4))
    plt.plot(sim_time[1::rk_interval-1], tire_force_log[4:7, 1:].T / 1000)
    plt.title("Rear Tire Forces")
    plt.legend(["$F_x$", "$F_y$", "$F_z$"])
    plt.ylabel("Force (kN)")
    plt.xlabel("Time (s)")
    plt.xlim([0, T_sim])
    plt.grid()
    plt.show()

    inter = 1
    Fx_f = tire_force_log[0, 1::inter].T / tire_force_log[2, 1::inter].T
    Fy_f = tire_force_log[1, 1::inter].T / tire_force_log[2, 1::inter].T

    Fx_r = tire_force_log[4, 1::inter].T / tire_force_log[6, 1::inter].T
    Fy_r = tire_force_log[5, 1::inter].T / tire_force_log[6, 1::inter].T

    plt.figure(figsize=(20, 4))
    plt.plot(sim_time[1::rk_interval-1], 1 - (Fx_f**2 + Fy_f**2) / param["mu_f"]**2 , label = "Front tire" )
    plt.plot(sim_time[1::rk_interval-1], 1 - (Fx_r**2 + Fy_r**2) / param["mu_r"]**2 , label = "Rear tire" )
    plt.title("Normalized Friction Cone")
    plt.xlabel("Time (s)")
    plt.xlim([0, T_sim])
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.plot(Fx_f, Fy_f , "-o",  label = "Front tire" )
    plt.plot(Fx_r, Fy_r , "-o", label = "Rear tire" )

    plt.plot(Fx_f[0], Fy_f[0] , "g*")
    plt.plot(Fx_r[0], Fy_r[0] , "g*")

    plt.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * param["mu_f"], np.sin(np.linspace(0, np.pi * 2, 100)) * param["mu_f"], "k--", label = "Front tire limits")
    plt.plot(np.cos(np.linspace(0, np.pi * 2, 100)) * param["mu_r"], np.sin(np.linspace(0, np.pi * 2, 100)) * param["mu_r"], "b--", label = "Rear tire limits")

    plt.title("Normalized Friction Cone")
    plt.ylabel("$F_y$")
    plt.xlabel("$F_x$")
    plt.legend()
    plt.grid()
    plt.show()