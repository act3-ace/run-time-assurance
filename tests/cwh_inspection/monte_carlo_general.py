"""
Monte Carlo Script for RTA tests

The script takes inputs specified below and runs trials for verification.

Input arguments:

1) constraint_keys - keywords for constraints to test ([] if you want to test all constraints)
2) dt              - timestep in seconds
3) time            - total simulation time in seconds
4) dist_params     - array of parameters for the distributions (ie. for gaussian will be mu and sigma respectively)
5) state_PDFs      - array for the distributions corresponding to each state
6) num_samples     - number of samples per state

"""
import csv
from fileinput import filename
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from run_time_assurance.zoo.cwh.inspection_1v1 import Inspection1v1RTA
import test_inspection_1v1
startTime = time.time()

############################################## ENVIRONMENT AND INITIAL CONDITIONS ##############################################

dt = 1 
total_time = 100
total_points = 1000

# [x, y, z, xdot, ydot, zdot, sun_angle, m_f]

l_b_position, u_b_position = [-1000, -1000, -1000], [1000, 1000, 1000]
l_b_velocity, u_b_velocity = [-1, -1, -1], [1, 1, 1]
l_b_sun_mass, u_b_sun_mass = [0, 0], [2*np.pi, 0.25]

l_bounds = l_b_position + l_b_velocity + l_b_sun_mass
u_bounds = u_b_position + u_b_velocity + u_b_sun_mass
rta = Inspection1v1RTA()
constraint_keys = []
env = test_inspection_1v1.Env(rta=rta, dt=dt, time=total_time, constraint_keys=constraint_keys)


############################################## ENVIRONMENT AND INITIAL CONDITIONS ##############################################

def randomMC(env,total_pts,l_bounds,u_bounds):
    
    tf_array = []
    count, t_count, curr_idx = 0, 0, 0

    sampler = qmc.LatinHypercube(d=8)
    pt_multiplier = 3
    sample = sampler.random(n=total_points * pt_multiplier)
    all_ics = qmc.scale(sample, l_bounds, u_bounds)

    for i in range(0,total_pts):
        rta_assured_safety_bool = None
        while rta_assured_safety_bool == None:
            initial_state = all_ics[curr_idx]
            init_temp = np.copy(initial_state)
            rta_assured_safety_bool = env.run_mc(initial_state)
            curr_idx += 1

        count += 1
        
        if (((count/total_pts)*100) % 10) == 0:
            print(str(round((count/total_pts)*100,4)),'% complete')
        
        if rta_assured_safety_bool:
            t_count += 1

        # Saves true or false or none associated with data point of IC's
        tf_array.append([rta_assured_safety_bool,init_temp.tolist()])
    
    return tf_array, t_count, count

############################################## CALL METHODS & SET FLAGS ##############################################


tf_array, t_count, count = randomMC(env,total_points,l_bounds,u_bounds)

print('Success Rate: ', str((t_count/count)*100), '%')
executionTime = time.time() - startTime
print("Execution time in hours: " + str(round(executionTime/3600,4)))

filename = ('MCSim_hypercube_numPts=' + str(total_points) + '_Date_' + datetime.today().strftime('%Y-%m-%d'))
with open((filename + ".csv"),"w+") as my_csv:
    newarray = csv.writer(my_csv,delimiter=',')
    newarray.writerows(tf_array)
