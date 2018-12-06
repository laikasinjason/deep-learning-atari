from IPython.core.debugger import set_trace
class Agent:
    # Class to save states

    def __init__(self):
        self.start_states = []
        self.end_states = []
        self.is_terminate = False
        self.lose_life = False
        self.total_reward = 0
        self.env_current_lives = 5

        
    def reset_env(self, env, pre_process):
        # First learning trial
        frame = env.reset()
        state = pre_process.preprocess(frame)
        self.start_states = [state, state, state, state]
        self.end_states = self.start_states
        self.total_reward = 0
        self.env_current_lives = env.env.ale.lives()
        self.is_terminate = False
        self.lose_life = False
        
    def do_action(self, env, pre_process, action):
        total_reward = 0
        self.lose_life = False
        # end_state is start_state1
        frame, reward, is_terminate, live_info = env.step(action)
        state = pre_process.preprocess(frame)
        total_reward = total_reward + reward
        
        if is_terminate:
            self.lose_life = True
            
        if 'ale.lives' in live_info:
            if live_info['ale.lives'] < self.env_current_lives:
                self.env_current_lives = live_info['ale.lives']
                self.lose_life = True

#         frame, reward, is_terminate, _ = env.step(action)
#         state2 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
#         frame, reward, is_terminate, _ = env.step(action)
#         state3 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
#         frame, reward, is_terminate, _ = env.step(action)
#         state4 = pre_process.preprocess(frame)
#         total_reward = total_reward + reward
        
        self.start_states = self.end_states.copy()
        self.end_states = self.end_states.copy()
        self.end_states.pop(0)
        self.end_states.append(state)
        self.is_terminate = is_terminate
        self.total_reward = pre_process.transform_reward(total_reward)