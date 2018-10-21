from IPython.core.debugger import set_trace
class Agent:
    # Class to save states

    def __init__(self):
        self.start_states = []
        self.end_states = []
        self.is_done = False
        self.total_reward = 0
        self.iteration = 0 # used for recording of iteration

        
    def reset_env(self, env, pre_process):
        # First learning trial
        frame = env.reset()
        state = pre_process.preprocess(frame)
        self.start_states = [state, state, state, state]
        self.end_states = self.start_states
        self.total_reward = 0
        self.is_done = False
        
    def do_action(self, env, pre_process, action):
        total_reward = 0
        # end_state is start_state1
        frame, reward, is_done, _ = env.step(action)
        state = pre_process.preprocess(frame)
        total_reward = total_reward + reward

#         frame, reward, is_done, _ = env.step(action)
#         state2 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
#         frame, reward, is_done, _ = env.step(action)
#         state3 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
#         frame, reward, is_done, _ = env.step(action)
#         state4 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
        self.start_states = self.end_states.copy()
        self.end_states.pop(0)
        self.end_states.append(state)
        self.is_done = is_done
        self.total_reward = pre_process.transform_reward(total_reward)