# %matplotlib inline
import matplotlib.pyplot as plt


def get_epsilon_for_iteration(current_iteration, stable_iteration = 1000000 , initial_epsilon = 1, end_epsilon=0.1):
    '''
    decrease the epsilon linearly from 1 to 0.1 over the first million frames, and fixed at 0.1 thereafter
    '''

    epsilon = end_epsilon

    if current_iteration <= stable_iteration:
        decrease_per_epsilon = (initial_epsilon - end_epsilon) / stable_iteration
        epsilon = initial_epsilon - current_iteration * decrease_per_epsilon

    return epsilon


if __name__ == '__main__':
    epsilons = []
    for i in range(2000000):
        epsilons.append(get_epsilon_for_iteration(i))

    plt.plot(epsilons)
    plt.ylabel('epsilon')
    plt.xlabel('time step')
    plt.show()