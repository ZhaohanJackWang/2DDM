import time
import numpy as np
import pandas as pd
from scipy.optimize import shgo

from param_obj import Objective

def Simulate(parameters, k, observation, lane_boundaries, verbose):
    """ Simulate trajectories based on MPC """

    # driver parameters
    s0 = 2
    T = 0.6 # JOHANSSON & RUMAR
    l0,w0 = 0., 0.
    discount = 1

    # time parameters
    time_interval = 0.4
    horizon_interval = 0.2
    prediction_horizon = 5 # np
    control_horizon = 1 # nc

    t = 0
    backup_states = pd.DataFrame()
    states = pd.DataFrame()
    k1_controls = pd.DataFrame()
    k2_controls = pd.DataFrame()

    veh_first_frame = observation.groupby('id').first().sort_values('frame_old').reset_index()

    for frame in observation.frame_old.unique():
        if verbose:
            t0 = time.time()

        t += time_interval
        t = round(t, 5)
        current_frame = veh_first_frame[veh_first_frame['frame_old'] == frame][['id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane', 'truck']]
        current_frame = current_frame if backup_states.empty else pd.concat([current_frame, backup_states[(backup_states['time'] == round(t-time_interval, 5)) & (backup_states['x'] < 400)][['id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane', 'truck']]])
        current_frame_backup = current_frame.copy()
        current_frame_backup['time'] = t
        current_frame_backup['frame_old'] = frame
        states = pd.concat([states, current_frame_backup[['time', 'frame_old', 'id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane', 'truck']]])

        for idx, row in current_frame.iterrows():
            # parameters
            ssd = parameters[parameters['id'] == row.id].iloc[0]['ssd']
            T_safe = parameters[parameters['id'] == row.id].iloc[0]['t_safe']
            alphas = parameters[parameters['id'] == row.id].loc[:, 'w1':'w6'].to_numpy().squeeze()
            
            obj = Objective([0,0,0,0], [0,0], np.array([[0,0,0,0,0,0,0]]), ssd, s0, T, T_safe, l0, w0, alphas, discount, horizon_interval, lane_boundaries, prediction_horizon, control_horizon, row.truck)
            
            veh = np.array(row[:-1])
            veh_ = np.array(current_frame[~(current_frame == row).all(axis=1)].drop(columns=['truck']))

            u_i_sequence = np.zeros((len(veh_), control_horizon, 2))

            z = np.array([veh]*(prediction_horizon+1))
            z_i = np.array([veh_]*(prediction_horizon+1))

            # calculate optimal control sequence
            opt_res = shgo(lambda u_sequence: obj.objective(u_sequence, z, z_i, u_i_sequence, t), bounds=[[obj.a_min,obj.a_max],[obj.delta_min,obj.delta_max]]*control_horizon)
            u_sequence = opt_res.x.reshape((control_horizon, 2))

            # update recorder
            k1_controls = pd.concat([k1_controls, pd.DataFrame([t, frame, row.id] + u_sequence[0].tolist(), index=['time', 'frame_old', 'id', 'a', 'delta']).T])
            if k == 1:
                backup_states = pd.concat([backup_states, pd.DataFrame([t, frame] + obj.rk4_step(z[0], u_sequence[0], t, time_interval).tolist() + [row.truck], index=['time', 'frame_old', 'id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane', 'truck']).T])

        # K=2
        if k == 2:
            for idx, row in current_frame.iterrows():
                # parameters
                ssd = parameters[parameters['id'] == row.id].iloc[0]['ssd']
                T_safe = parameters[parameters['id'] == row.id].iloc[0]['t_safe']
                alphas = parameters[parameters['id'] == row.id].loc[:, 'w1':'w6'].to_numpy().squeeze()
                
                obj = Objective([0,0,0,0], [0,0], np.array([[0,0,0,0,0,0,0]]), ssd, s0, T, T_safe, l0, w0, alphas, discount, horizon_interval, lane_boundaries, prediction_horizon, control_horizon, row.truck)
            
                veh = np.array(row[:-1])
                veh_ = np.array(current_frame[~(current_frame == row).all(axis=1)].drop(columns=['truck']))

                u_i_sequence = k1_controls[k1_controls['frame_old']==frame].set_index('id').loc[veh_[:, 0]][['a', 'delta']]
                u_i_sequence = np.array(u_i_sequence).reshape(len(veh_), control_horizon, 2)

                z = np.array([veh]*(prediction_horizon+1))
                z_i = np.array([veh_]*(prediction_horizon+1))

                # calculate optimal control sequence
                opt_res = shgo(lambda u_sequence: obj.objective(u_sequence, z, z_i, u_i_sequence, t), bounds=[[obj.a_min,obj.a_max],[obj.delta_min,obj.delta_max]]*control_horizon)
                u_sequence = opt_res.x.reshape((control_horizon, 2))

                # update recorder
                k2_controls = pd.concat([k2_controls, pd.DataFrame([t, frame, row.id] + u_sequence[0].tolist(), index=['time', 'frame_old', 'id', 'a', 'delta']).T])
                backup_states = pd.concat([backup_states, pd.DataFrame([t, frame] + obj.rk4_step(z[0], u_sequence[0], t, time_interval).tolist() + [row.truck], index=['time', 'frame_old', 'id', 'x', 'y', 'v', 'psi', 'width', 'height', 'lane', 'truck']).T])

        if verbose:
            t1 = time.time()
            print(t, (t1-t0))

    states['time'] = states['frame_old'] * 0.04

    if k == 1:
        controls = k1_controls
    if k == 2:
        controls = k2_controls
    return states, controls