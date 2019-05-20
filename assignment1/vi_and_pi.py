### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):

        # initialize old value function for all states
        value_function = np.zeros(nS)
        
        # Loop
        while True:
                print('inside first loop \n')
                
                # delta gets zero
                delta = 0

                # Loop for each state to perform a full back-up
                for s in range(nS):
                        print('state:', s)

                        # set value to zero
                        v = 0
                        
                        # ------------ Deviation from 4.1 algorithm ------------ #
                        # Loop through set of possible next actions

                        print('TYPES: \n')
                        print(type(a))
                        print(type(action_prob))
                        print('\n')
                        
                        for a, action_prob in enumerate(policy[s]):

                                # For each action, look at its possible next state
                                for prob, next_state, reward, done in P[s][a]:

                                        # Calculate the expected value using equation 4.6
                                        v += action_prob * prob * (reward +
                                                                   discount *
                                                                   V[next_state])
                        # ------------ Deviation from 4.1 algorithm ------------ #

                        # Compute the change in value functions across states
                        delta = max(delta, np.abs(v - V[s]))
                        V[s] = v

                        print('delta: d%', delta)

                # Terminate policy evaluation when the change in value function is below threshold
                if delta < theta:
                        break

        # Final value function
        value_function = np.array(V)
        
        return value_function



if __name__ == "__main__":

        # comment/uncomment these lines to switch between deterministic/stochastic envs
       	env = gym.make("Deterministic-4x4-FrozenLake-v0")

       	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

        # comment/uncomment to test policy evalution only
        policy = np.zeros(env.nS, dtype=int )
        action_value_function = policy_evaluation(env.P, env.nS, env.nA, policy, gamma=0.9, tol=1e-3)

        
        #V_pi, p_pi = policy_iteration
