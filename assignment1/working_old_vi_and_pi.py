import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):

    print('policy evaluation begins \n')
    
    value_function = np.zeros(nS)
    
    while True:
        delta = 0

        print('Inside policy eval first loop \n')

        
        #For each state, perform a full backup
        for s in range(nS):
            v = 0

            print('type of policy[s]: ', type(policy[s])) # THIS IS THE WRONG TYPE!!!!
            
            
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
            #for a in policy[s]:
                

            
                # For each action, look at the possible next states
                for prob, next_state, reward, done in P[s][a]:
                    
                    # Calculate the expected value, eqtn 4.6
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
                    
            # Compute change in value functions across states 
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
                    
            print('delta: %d', delta)
                    
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
            
    value_function = np.array(V)
            
    return value_function


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):

    print('nS: ', nS)
    
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    
    action_value_function = policy_evaluation(env.P, env.nS, env.nA, policy, gamma=0.9, tol=1e-3)
    
    return value_function, policy



if __name__ == "__main__":

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    env = gym.make("Deterministic-4x4-FrozenLake-v0")
    # env = gym.make("Stochastic-4x4-FrozenLake-v0")

    print('TYPE: \n')
    print(type((env.P)))
    print(env.P.keys())
    print(env.P[0].keys())
    
    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    #render_single(env, p_pi, 100)
