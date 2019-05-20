import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):

    print('policy evaluation begins \n')
    
    V = np.zeros(nS)
    value_fcn_counter = 0
    
    while True:
        delta = 0

        #For each state, perform a full backup
        for s in range(nS):
            v = 0
            
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
            
                # For each action, look at the possible next states
                for prob, next_state, reward, done in P[s][a]:
                    
                    # Calculate the expected value, eqtn 4.6
                    v += action_prob * prob * (reward + gamma * V[next_state])
                    
            # Compute change in value functions across states 
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
                    
            print('delta: %d', delta)
                    
        # Stop evaluating once our value function change is below a threshold
        if delta < tol:
            break
        value_fcn_counter += 1
        print('Value Function Identified! \n')

    print('Number of Policy Iterations: ', value_fcn_counter)
    value_function = np.array(V)
            
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):

    new_policy = np.zeros(nS, dtype='int')

    # policy-stable <-- true
    policy_stable = True
    
    # Loop for each state in the state set
    for s in range(nS):

        # old-action <-- pi[s]; take the best action under the current policy
        old_action = np.argmax(policy[s])
        #old_action = policy[s]

        # set the maximum value and best action 
        maxvsa = -1 
        maxa = -1
        
        # Loop through the next possible actions for the current state to find
        # the best action by one-step lookahead (ties resolved arbitratily)
        # pi[s] <-- argmax_a sum_s'_r p(s',r|s,a)[r + gammaV(s')]
        for a in range(nA):

            vsa = 0

            # Loop through the possible next states for each action 
            #for next_state in range(P[s][a]):
            for prob, next_state, reward, done in P[s][a]:
                
                prob_action = prob
                current_reward = reward
                future_reward = gamma * value_from_policy[next_state]

                vsa += prob_action * (current_reward + future_reward)

                #A[a] += prob * (reward + gamma + V[next_state])

            #return a
                
                
            # if the current value fcn is greater than the best value fcn,
            # update the maximum value fcn and the best action
            if vsa > maxvsa:

                # the max value fcn 
                maxvsa = vsa
                
                # best action
                maxa = a

        # if the old-action isn't equal to
        # pi(s) then policy-stable <-- false
        if old_action != maxa:
            policy_stable = False

        # if policy-stable then stop and return V~v* amd pi~pi*, else go to policy evaluation  
        if policy_stable:
            policy[s] = maxa

        return policy
                
        


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):

    print('nS: ', nS)
    
    value_function = np.zeros(nS)
    #policy = np.zeros(nS, dtype=int)

    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    action_value_function = policy_evaluation(env.P, env.nS, env.nA, policy, gamma=0.9, tol=1e-3)
    print ('action value function:\n ', action_value_function)

    policy = policy_improvement(P, nS, nA, action_value_function, policy, gamma=0.9)
    print('policy:\n', policy)
    
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
