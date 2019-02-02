from keras import backend as K
import keras
import random
import numpy as np


def fit_batch(model, target_model, start_states, actions, rewards, next_states, is_terminate, pre_process, 
              gamma = 0.99, learning_rate = 0.1):
    """Do one deep Q learning iteration.
    
    Params:
    - model: The DQN
    - target_model: The target DQN
    - gamma: Discount factor (should be 0.99)
    - start_states: numpy array of starting states
    - actions: numpy array of one-hot encoded actions coend_staterresponding to the start states
    - rewards: numpy array of rewards corresponding to the start states and actions
    - next_states: numpy array of the resulting states corresponding to the start states and actions
    - is_terminate: numpy boolean array of whether the resulting state is terminal
    
    """
    # convert to a 4-d array
    next_states = np.stack(next_states).swapaxes(1,2).swapaxes(2,3)
    start_states = np.stack(start_states).swapaxes(1,2).swapaxes(2,3)
    next_states = pre_process.to_float(next_states)
    start_states = pre_process.to_float(start_states)
            
    # Run one fast-forward to get the Q-values for all actions
    target = model.predict(start_states)
    current_Q_values = target[actions.astype(bool)]
    
    # Predict the Q values of the next states. Passing ones as the mask.
#     next_Q_values = target_model.predict([next_states,  np.ones(actions.shape)])
    next_Q_values = target_model.predict(next_states)
    
    # The Q values of the terminal states is 0 by definition, so override them
    next_Q_values[is_terminate.astype(bool)] = 0
    # The Q values of each start state is the reward + gamma * the max next state Q value
    next_Q_values = rewards + gamma * np.max(next_Q_values, axis=1)
    new_Q_values = (1-learning_rate) * current_Q_values + learning_rate * next_Q_values
    # Fit the keras model. Pass the actions as the mask and multiplying
    # the targets by the actions.
    
    # Set the new Q values to target
    target[actions.astype(bool)] = new_Q_values
    
    loss = model.fit(start_states, target,
        epochs=1, batch_size=len(start_states), verbose=0
    )
    return loss
    
    
def get_epsilon_for_iteration(current_iteration, stable_iteration = 1000000 , initial_epsilon = 1, end_epsilon=0.1):
    '''
    decrease the epsilon linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    '''
    epsilon = end_epsilon

    if current_iteration <= stable_iteration:
        decrease_per_epsilon = (initial_epsilon - end_epsilon) / stable_iteration
        epsilon = initial_epsilon - current_iteration * decrease_per_epsilon

    return epsilon

def choose_best_action(model, states, no_actions):
    # swap axes to archieve keras model input
    state = np.array(states).swapaxes(0,1).swapaxes(1,2)
    # extend to a 4-d array
    state = np.expand_dims(state, axis=0)
#     return model.predict([state, np.expand_dims(np.ones(no_actions), axis=0)]).argmax()
    return model.predict(state).argmax()

def q_iteration(env, model, target_model, agent, iteration, ring_buf, one_hot_encoder, pre_process):
    # Choose epsilon based on the iteration
    epsilon = get_epsilon_for_iteration(iteration)

    # Choose the action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, agent.end_states, env.action_space.n)

    # Play one game iteration (action X 4 times)
    agent.do_action(env, pre_process, action)
    ring_buf.append((agent.start_states, action, agent.total_reward, agent.end_states, agent.lose_life))

    # Sample and fit
    batch = ring_buf.sample_batch(32)
    loss = fit_batch(model, target_model, batch[0], one_hot_encoder.transform(batch[1].reshape(-1,1)), 
              batch[2], batch[3], batch[4], pre_process)
    return loss


def evaluate(logger, model, agent, pre_process, env):

    print("Evaluation started.")
    for i in range(0, logger.evaluation_number):
        agent.reset_env(env, pre_process)

        while not agent.is_terminate:
            action = choose_best_action(model.model, agent.end_states, env.action_space.n)
            agent.do_action(env, pre_process, action)
            logger.write(agent)