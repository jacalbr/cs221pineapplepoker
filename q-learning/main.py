from util import PipoMDP, DummyQLearning, TabularQLearning, DeepQLearning

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')[:-window_size]


# Comment out all but one!
rl = DummyQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
rl = TabularQLearning(mdp.actions, mdp.discount, explorationProb=0.15)
rl = DeepQLearning(mdp.actions, mdp.discount, explorationProb=0.15)


mdp = PipoMDP()
totalRewards, end_score = simulate(mdp, rl, train=True, numTrials=10000, verbose=True, demo=False)
plt.plot(movingaverage(np.array(totalRewards), 200))
