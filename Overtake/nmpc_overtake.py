import casadi as ca
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import time

def nmpc_controller():
    # Declare simulation constants
    T = 5.0 # planning horizon in seconds, controls how far into the future we plan
    N = 50  # number of control intervals, defines the resolution of our control actions (h = 0.1s)
    h = T / N

    # system dimensions
    Dim_state = 4 # Number of states: x-position, y-position, orientation, speed
    Dim_ctrl  = 2 # Number of control inputs: acceleration and steering angle

    # additional parameters
    x_init = ca.MX.sym('x_init', (Dim_state, 1))  # initial condition, # the state should be position to the leader car
    v_leader = ca.MX.sym('v_leader',(2, 1))       # leader car's velocity w.r.t ego car
    v_des = ca.MX.sym('v_des')                    # desired speed for the ego vehicle
    delta_last = ca.MX.sym('delta_last')          # last steering angle (used for rate constraints)
    params = ca.vertcat(x_init, v_leader, v_des, delta_last)
    
    # Continuous dynamics model
    x_model = ca.MX.sym('xm', (Dim_state, 1))
    u_model = ca.MX.sym('um', (Dim_ctrl, 1))

    L_f = 1.0 # Car parameters, do not change
    L_r = 1.0 # Car parameters, do not change

    beta = ca.atan(L_r / (L_r + L_f) * ca.atan(u_model[1])) # The angle at which the car moves sideways relative to its orientation

    # xdot describes how each state variable changes over time based on current state and control (x-position change, y-position change, orientation change, speed change)
    xdot = ca.vertcat(x_model[3] * ca.cos(x_model[2] + beta) - v_leader[0],
                      x_model[3] * ca.sin(x_model[2] + beta) - v_leader[1],
                      x_model[3] / L_r * ca.sin(beta),
                      u_model[0])

    # Discrete time dynmamics model
    Func_dynmaics_dt = ca.Function('f_dt', [x_model, u_model, params], [xdot * h + x_model])
    
    # Declare model variables, note the dimension
    x = ca.MX.sym('x', (Dim_state, N + 1))
    u = ca.MX.sym('u', (Dim_ctrl, N))

    # Define the cost function (objective) components
    # These encourage the car to stay in its lane, follow the leader, and achieve desired speed
    # Running cost keeps a light lane weight so the car is free to pull out and pass,
    # while the terminal cost strongly pulls it back to y = 0 with zero yaw at v_des.
    P = 10. * x_model[1]**2 + 1000. * x_model[2]**2 + 100. * (x_model[3] - v_des)**2
    L = 0.01 * x_model[1]**2 + 1. * x_model[2]**2 + 10. * (x_model[3] - v_des)**2 \
        + 0.01 * u_model[0]**2 + 1. * u_model[1]**2

    Func_cost_terminal = ca.Function('P', [x_model, params], [P])
    Func_cost_running = ca.Function('Q', [x_model, u_model, params], [L])

    # state and control constraints
    state_ub = np.array([ 1e4,  1e2,  1e2,  1e2]) # large bounds for position, tighter on lateral position
    state_lb = np.array([-1e4, -1e2, -1e2, -1e2])
    ctrl_ub  = np.array([  4.,  0.6]) # Control limits for acceleration and steering angle
    ctrl_lb  = np.array([-10., -0.6])
    
    # upper bound and lower bound
    ub_x = np.matlib.repmat(state_ub, N + 1, 1)
    lb_x = np.matlib.repmat(state_lb, N + 1, 1)

    ub_u = np.matlib.repmat(ctrl_ub, N, 1)
    lb_u = np.matlib.repmat(ctrl_lb, N, 1)

    ub_var = np.concatenate((ub_u.reshape((Dim_ctrl * N, 1)), ub_x.reshape((Dim_state * (N+1), 1))))
    lb_var = np.concatenate((lb_u.reshape((Dim_ctrl * N, 1)), lb_x.reshape((Dim_state * (N+1), 1))))

    # dynamics constraints: x[k+1] = x[k] + f(x[k], u[k]) * dt
    # This enforces the system's discrete dynamics, meaning each next state is based on the current state and control.
    cons_dynamics = []
    ub_dynamics = np.zeros((Dim_state * N, 1))
    lb_dynamics = np.zeros((Dim_state * N, 1))
    for k in range(N):
        # Fx represents the calculated state at the next time step based on the dynamics model.
        # For each state variable (e.g., x-position, y-position, orientation, speed), we add a constraint.
        # This loop means that the computed next state (Fx) matches the predicted state (x[:, k+1]).
        Fx = Func_dynmaics_dt(x[:, k], u[:, k], params)  
        for j in range(Dim_state):
            cons_dynamics.append(x[j, k+1] - Fx[j])


    # state constraints: G(x) <= 0
    cons_state = []
    for k in range(N):
        #### collision avoidance:
        cons_state.append(1 - (x[0, k+1] / 30.)**2 - (x[1, k+1] / 2.)**2)

        #### Maximum lateral acceleration ####
        dx = (x[:, k+1] - x[:, k]) / h  # Change in state over time step
        ay = dx[2] * x[3, k] # lateral acc (change in orientation * speed)
        
        gmu = (0.5 * 0.6 * 9.81)
        # Upper and lower bound on lateral acceleration
        cons_state.append(ay - gmu)
        cons_state.append(-ay - gmu)

        #### lane keeping ####
        # Upper and lower bound on lateral position
        cons_state.append(x[1, k+1] - 3.)
        cons_state.append(-1. - x[1, k+1])

        #### steering rate ####
        if k >= 1:
            d_delta = u[1, k] - u[1, k-1] # Difference between current and previous steering angle

            # Constraint steering rate to ensure smooth changes, scaled by time step `h` for discretization.
            # Upper and lower bound on steering rate
            cons_state.append(d_delta - 0.6 * h)
            cons_state.append(-d_delta - 0.6 * h)
        else:
            d_delta = u[1, k] - delta_last # for the first input, given d_last from param
            cons_state.append(d_delta - 0.6 * h)
            cons_state.append(-d_delta - 0.6 * h)

    ub_state_cons = np.zeros((len(cons_state), 1))
    lb_state_cons = np.zeros((len(cons_state), 1)) - 1e5

    # cost function: # NOTE: You can also hard code everything here
    J = Func_cost_terminal(x[:, -1], params)
    for k in range(N):
        J = J + Func_cost_running(x[:, k], u[:, k], params)

    # initial condition as parameters
    cons_init = [x[:, 0] - x_init]
    ub_init_cons = np.zeros((Dim_state, 1))
    lb_init_cons = np.zeros((Dim_state, 1))
    
    # Define variables for NLP solver
    vars_NLP   = ca.vertcat(u.reshape((Dim_ctrl * N, 1)), x.reshape((Dim_state * (N+1), 1)))
    cons_NLP = cons_dynamics + cons_state + cons_init
    cons_NLP = ca.vertcat(*cons_NLP)
    lb_cons = np.concatenate((lb_dynamics, lb_state_cons, lb_init_cons))
    ub_cons = np.concatenate((ub_dynamics, ub_state_cons, ub_init_cons))

    # Create an NLP solver
    prob = {"x": vars_NLP, "p":params, "f": J, "g":cons_NLP}
    
    return prob, N, vars_NLP.shape[0], cons_NLP.shape[0], params.shape[0], lb_var, ub_var, lb_cons, ub_cons
