class Logger:
    
    # Class for saving the performance indicators

    def __init__(self, evaluation_number):
        self.sum_ret = 0.0 # reward in one play
        self.total_ret = 0.0 # total reward in all plays
        self.evaluation_number = evaluation_number
        self.runs_in_eval = 0
        self.num_evals = 1 # counter
        self.max_return = 0
        self.min_return = 999999.9

    def write(self, agent):
        self.sum_ret += agent.total_reward
        self.total_ret += agent.total_reward
        if agent.is_done:
            print ("Evaluation " + str(self.runs_in_eval) + " :" + str(self.sum_ret))
            if self.sum_ret > self.max_return:
                self.max_return = self.sum_ret
            if self.sum_ret < self.min_return:
                self.min_return = self.sum_ret
            self.runs_in_eval += 1
            self.sum_ret = 0.0

            if self.runs_in_eval == self.evaluation_number:
                self.runs_in_eval = 0
                self.num_evals += 1
                avg = float(self.total_ret)/float(self.evaluation_number)
                print("Max reward : " + str(self.max_return) + "Min reward: " + str(self.min_return)\
                      + "Average reward in current episode: " + str(avg))
                f = open("model_learn_progress.txt", "a")
                f.write(str(agent.iteration)+","+str(avg)+","+str(self.max_return)+","\
                        +str(self.min_return)+"\n")
                self.total_ret = 0.0
                self.max_return = 0.0
                self.min_return = 999999.9