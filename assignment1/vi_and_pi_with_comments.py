### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

"""For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters env, P, nS, nA, gamma are defined as follows:

       P: nested dictionary
		From gym.core.Environment, represents the transition probabilities of the environment.
		For each pair of states in [1, nS] and 
                                 actions in [1, nA],
                                 P[state][action] is a
		a list of transition tuples (probability, nextstate, reward, terminal) where
			- probability: float
				probability of transitioning from "state" to "nextstate" w/ "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
        """Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]; [S, A] shaped matrix representing the policy.
		The policy to evaluate. Maps states to actions.
	tol: float; theta determining the accuracy of estimation
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

        value_function = np.zeros(nS)


        ############################
	# YOUR IMPLEMENTATION HERE #


        # Pseudocode Algorithm Section 4.1

        # Initialize Algorithm Values 
        # input pi - the policy to be evaluated --> passed in as fcn argument, "policy" 
        # threshold (theta) - determines accuracy of estimation --> passed in as fcn arg., "tol"
        
        # initialize old value function for all states
        value_function = np.zeros(nS)

        # Loop
        while True:

                print('inside first loop /n')
                
                # delta gets zero
                delta = 0

                # Loop for each state to perform a full back-up
                for s in range(nS):
                        
                        # set value to zero
                        v = 0
                
                        # ------------ Deviation? ------------ #
                        # Loop through set of possible next actions
                        for a, action_prob in enumerate(policy[s]):

                                # For each action, look at its possible next state
                                for prob, next_state, reward, done in P[s][a]:

                                        # Calculate the expected value using equation 4.6
                                        v += action_prob * prob * (reward +
                                                                   discount *
                                                                   V[next_state])
                        # ------------ Deviation? ------------ #

                        # Compute the change in value functions across states
                        delta = max(delta, np.abs(v - V[s]))
                        V[s] = v

                        print('delta: d%', delta)

                # Terminate policy evaluation when the change in value function is below threshold
                if delta < theta:
                        break

        # Final value function
        value_function = np.array(V)
                                                                   
        ############################

        return value_function
