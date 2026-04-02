import numpy as np
from random import seed as py_seed
import math
import utils
import copy

seed = 100
np.random.seed(seed)
py_seed(seed)

target_pos1 = np.array([50.0, 50.0])  
target_pos2 = np.array([50.0, -50.0])  

class RA():
    def __init__(self):
        self.Ns = 120
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 60
        self.h_ext = None
        self.beta = 4000
        self.sigma_ang = 5*np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins = np.random.choice([1,0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = False
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 4))

ring = RA()
L = 1000

def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

previous_distance1 =1000000
previous_distance2 =1000000

pos = np.zeros((L, 2))
bump_angles = np.zeros(L)
spins_history = np.zeros((L, ring.Ns), dtype=int)
heading = 0.0

# for target tracking
distance_between_target = np.zeros(L)

# for animation
hext_for_time = []
target_for_time = np.zeros(L)
target_reached_L = -1

for t in range (L):
    dist_to_target1 = euclidean_distance(pos[t], target_pos1)
    dist_to_target2 = euclidean_distance(pos[t], target_pos2)
    if (dist_to_target1 > previous_distance1) and (dist_to_target2 > previous_distance2):
        target_reached_L = t
        break
    # if (dist_to_target2 > previous_distance1):
    #     target_reached_L = t
    #     break
    # print(dist_to_target1,dist_to_target2)
    previous_distance1 = dist_to_target1
    previous_distance2 = dist_to_target2
    theta_target1 = np.arctan2( target_pos1[1] - pos[t][1],target_pos1[0] - pos[t][0])
    theta_target2 = np.arctan2( target_pos2[1] - pos[t][1],target_pos2[0] - pos[t][0])
    if ring.allocentric:
        theta_diff1 = theta_target1
        theta_diff2 = theta_target2
    else:
        theta_diff1 = (theta_target1-heading) % (2 * np.pi) 
        theta_diff2 = (theta_target2-heading) % (2 * np.pi)

    # if t%10 == 0 and t>1:    
    #     utils.compute_h_ext_multiple(ring,[ theta_diff2])
    # else:   
    #     utils.compute_h_ext_multiple(ring,[theta_diff1, theta_diff2])
        

    utils.compute_h_ext_multiple(ring,[theta_diff1, theta_diff2])

    #for animation
    hext_for_time.append(ring.h_ext)
    target_for_time[t] = theta_target1 # incorrect for two targets, but leave as is for simplicity

    print(" length of h_ext: ",len(ring.h_ext))
        
    # utils.plot_any_line([math.degrees(theta) for theta in ring.thetas],ring.h_ext, 'Neuron index', 'h_i','External Sensory Input (h_i) vs Neuron index')
    
    for _ in range (ring.updates_per_step):
        i = np.random.randint(0,ring.Ns)
        delta_H = utils.compute_delta_H(ring,i)

        if delta_H < 0:
            # Energy difference negative — always accept
            ring.spins[i] = 1 - ring.spins[i]
        else:
            # Energy difference positive — accept with probability exp(-beta * delta_H)
            p = np.exp(-ring.beta * delta_H)
            if np.random.rand() < p:
                ring.spins[i] = 1 - ring.spins[i]
                
    active_indices = np.where(ring.spins == 1)[0]
    n_active = len(active_indices)
    if n_active > 0:
        phi = np.angle(np.sum(np.exp(1j * ring.thetas[ring.spins == 1])))
    else:
        phi = 0.0  
    bump_angles[t] = math.degrees(phi)

    print(f'Time step {t+1}/{L}, Position: {pos[t]}, theta_targets: {[math.degrees(ayyo) for ayyo in [theta_target1, theta_target2]]} degrees, Bump angle: {bump_angles[t]:.2f} degrees, num_step: {ring.updates_per_step}')

    
    if ring.allocentric:
        # allocentric motion
        if t < L-1:
            pos[t+1, 0] = pos[t, 0] + ring.v_0 * np.cos(phi)/ring.Ns
            pos[t+1, 1] = pos[t, 1] + ring.v_0 * np.sin(phi)/ring.Ns
    else:  
        # egocentric motion
        heading = (phi + heading) % (2 * np.pi) 
        if t < L-1:
            pos[t+1, 0] = pos[t, 0] + ring.v_0 * np.cos(heading)/ring.Ns
            pos[t+1, 1] = pos[t, 1] + ring.v_0 * np.sin(heading)/ring.Ns

    spins_history[t] = copy.deepcopy(ring.spins)

print("Simulation complete. ")

if target_reached_L == -1:
    utils.plot_summary_with_multiple_targets(ring,pos,spins_history,bump_angles,[target_pos1, target_pos2],L)
else:
    print("Target reached at time step: ",target_reached_L)
    utils.plot_summary_with_multiple_targets(ring,pos[:target_reached_L],spins_history[:target_reached_L],bump_angles[:target_reached_L],[target_pos1, target_pos2],target_reached_L)
