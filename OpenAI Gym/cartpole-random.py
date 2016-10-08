# submit: Did not solve the environment. Best 100-episode average reward was 136.13 +- 4.09. 
#(CartPole-v0 is considered "solved" when the agent obtains an average reward of at least 195.0 over 100 consecutive episodes.)
import gym
import numpy as np
import matplotlib.pyplot as plt
# get the total reward for one episode, given parameters;
def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    for _ in xrange(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)

    counter = 0
    bestparams = None
    bestreward = 0
    for _ in xrange(10000):
        counter += 1
        parameters = np.random.rand(4) * 2 - 1
        reward = run_episode(env,parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters # record the best value
            if reward >= 200:
                break

    if submit:
        for _ in xrange(100):
            run_episode(env,bestparams) # run the best value for submit
        env.monitor.close()

    return counter # the number of the episodes needed to run to get the parameters to keep the pole up for 200 timesteps(reward==200);

# train an agent to submit to openai gym
# train(submit=True)

# calculate the averge number of episodes needed;
results = []
for _ in xrange(1000):
    results.append(train(submit=False))
print np.sum(results) / 1000.0 #average the results to get the avage number of episodes to keep the pole up;

# create graphs
plt.hist(results,50,normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Episodes required to reach 200')
plt.ylabel('Frequency')
plt.title('Histogram of Random Search')
plt.show()
