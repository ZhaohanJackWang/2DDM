import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from scipy.stats import rankdata

from param_fit_traj import Simulate

def get_sv_state(id, frames, highD):
    """    Get the state of a subject vehicle in the highD dataset.  """

    # Get the state of the subject vehicle
    sv = highD[(highD['id'] == id)]
    sv['v'] = np.sqrt(sv['xVelocity']**2 + sv['yVelocity']**2)
    sv['y_s'] = sv['y'].shift(-1)
    sv['x_s'] = sv['x'].shift(-1)

    psi = [np.arctan((sv.iloc[1]['y']-sv.iloc[0]['y'])/(sv.iloc[1]['x']-sv.iloc[0]['x']+1e-6))]
    for idx, row in enumerate(sv.itertuples()):
        psi.append(0.04*row.v/row.width/2 * np.sin(np.arcsin((row.y_s - row.y)/0.04/row.v) - psi[-1]) + psi[-1])
    psi = [i for i in psi if str(i) != 'nan']
    sv['psi'] = psi

    # Fix dimensions of a vehicle
    sv_state = sv[['frame','id','x','y','v','psi','width','height']].reset_index(drop=True)
    sv_state['sv'] = 1
    sv_state['x'] = sv_state['x'] + sv_state['width']/2
    sv_state['y'] = sv_state['y'] + sv_state['height']/2
    sv_state.loc[:, 'y'] = old_lane_bounds[-1] - sv_state['y'] # reverse y direction

    # Sort frames
    sv_state = sv_state[np.isin(sv_state['frame'], frames)]
    sv_state['frame_old'] = sv_state['frame']
    sv_state.loc[:, 'frame'] = rankdata(sv_state['frame'], method='dense')
    return sv_state

def get_highd_track(track, laneID, time_interval, lane_boundaries):
    """    Get the highD track data for a specific track and lane IDs.  """
    
    # Load the highD dataset
    highD = pd.read_csv(f"HighD/{track}_tracks.csv")
    highD_meta = pd.read_csv(f'HighD/{track}_tracksMeta.csv')

    highD = highD.merge(highD_meta[['id', 'class']], left_on='id', right_on='id')
    highD['truck'] = np.where(highD['class'] == 'Truck', 1, 0)

    # Get leftward traffic
    lanes678 = highD[highD['laneId'].isin(laneID)]['id'].unique()

    startID = 0
    endID = len(lanes678)-1

    start_frame = highD[highD['id'] == lanes678[startID]].frame.min()
    end_frame = highD[highD['id'] == lanes678[endID]].frame.max()
    frames = np.arange(start_frame, end_frame, time_interval/0.04)

    # Create df for observations
    observation = pd.DataFrame()

    for vehID in lanes678[startID:endID]:
        sv_state = get_sv_state(vehID, frames, highD)
        observation = pd.concat([observation, sv_state])

    observation['lane'] = [np.digitize(np.array(observation)[i][3], lane_boundaries) for i in range(len(observation))]
    observation = observation.merge(highD[['id', 'truck']].drop_duplicates('id'), left_on='id', right_on='id')
    observation['time'] = observation['frame_old']*0.04
    return observation

if __name__ == '__main__':

    # Define track and lane parameters
    track = '50'
    laneID = [6, 7, 8, 9]
    old_lane_bounds = np.array([31.40,35.53,39.31,43.56])
    lane_boundaries = (old_lane_bounds[-1] - old_lane_bounds)[::-1]
    
    # Define simulation parameters
    k = 1
    time_interval = 0.4 # must be multiple of 0.04

    # Obtain base track data
    observation = get_highd_track(track, laneID, time_interval, lane_boundaries)

    # Define weights
    smc_particles_car = pd.read_csv('SMC_particles_car_new.csv', index_col=0)
    smc_particles_truck = pd.read_csv('SMC_particles_truck_new.csv', index_col=0)

    params = pd.DataFrame([np.hstack([observation.drop_duplicates('id')['id'].values[i], smc_particles_car.sample().to_numpy()[0]]) if observation.drop_duplicates('id')['truck'].values[i] == 0 
                           else np.hstack([observation.drop_duplicates('id')['id'].values[i], smc_particles_truck.sample().to_numpy()[0]]) for i in range(len(observation.drop_duplicates('id')))],
                           columns=['id', 'ssd', 't_safe', 'w1', 'w2', 'w3', 'w4', 'w5', 'w6'])
 
    # Simulate trajectories
    states, controls = Simulate(params, k, observation, lane_boundaries, verbose=True)

    # Save results
    states.rename(columns={"pred_x": "x", "pred_y": "y", "pred_v": "v", "pred_psi": "psi"}, inplace=True)
    
    if k == 1:
        states.to_csv(f'highD_results/k1/states.csv')
        controls.to_csv(f'highD_results/k1/controls.csv')

    else:
        states.to_csv(f'highD_results/states.csv')
        controls.to_csv(f'highD_results/controls.csv')