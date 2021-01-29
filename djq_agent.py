"""
The module 'djq_agent' define trading strategies, the agent tells what to do on your account when new info available.
- action is int, with 2 means 'long', 1 means 'keep', 0 means 'short'
- observation, composites of predict result provided by models, and position info of recent days
- param 'window' means the length of input_dims, so that observation = [predict result]*window + [position]*window
- ThresholdAgent learns the best up/down thresholds that maximize the profit, when the result is above up threshold,
the agent generate a long signal
- DqnAgent, DdqnAgent, CemAgent all trained by 'keras-rl', using deep reinforce learning method
- MultiAgent is an ensemble RL agent, set the list of sub-agent and number of each agent, it will collect every
sub-agent's result, and return the action of the most picked
"""
import djq_utils
from abc import ABCMeta,abstractmethod
import os, json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.agents import CEMAgent
from collections import Counter
import multiprocessing
import numpy as np


class Agent(metaclass=ABCMeta):
    BASE_DIR = os.path.abspath('.') + '/agent/'
    AGENT_NAME = 'Agent'

    def __init__(self, name, window=5):
        self.name = self.AGENT_NAME + '#' + name
        self.window = window
        _, self.model_name, self.etf_name, self.id = self.name.split('#')
        if not os.path.exists(self.BASE_DIR + self.name):
            self._train()
        self._load()

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _train(self):
        pass

    @abstractmethod
    def get_action(self, observation):
        pass


class ThresholdAgent(Agent):
    AGENT_NAME = 'ThresholdAgent'

    def _load(self):
        with open(self.BASE_DIR + self.name + '/config.json', 'r') as f:
            self.config = json.load(f)
        self.u_threshold = self.config['u_threshold']
        self.d_threshold = self.config['d_threshold']

    def _train(self):
        env = djq_utils.stock_env(self.model_name, self.etf_name, window=self.window)
        best_threshold_u = 0
        best_threshold_d = 0
        best_score = 1
        for threshold_u in range(100):
            for threshold_d in range(-100, threshold_u):
                res = djq_utils.Monte_Carlo_Simulation(env, threshold_u / 10, threshold_d / 10, mode='all')
                if res > best_score:
                    best_score = res
                    best_threshold_u = threshold_u / 10
                    best_threshold_d = threshold_d / 10
        print('The best threshold_u and threshold_d for {} is {} & {}, the profit is {}'.format(self.model_name,
                                                                                                best_threshold_u,
                                                                                                best_threshold_d,
                                                                                                best_score))
        os.mkdir(self.BASE_DIR + self.name)
        config = dict()
        config['u_threshold'] = best_threshold_u
        config['d_threshold'] = best_threshold_d
        with open(self.BASE_DIR + self.name + '/config.json', 'w') as f:
            json.dump(config, f)

    def get_action(self, observation):
        if observation[-1] >= self.u_threshold:
            return 2
        elif observation[-1] <= self.d_threshold:
            return 0
        else:
            return 1


class CemAgent(Agent):
    AGENT_NAME = 'CemAgent'

    def __init__(self, name, window=5):
        self.window = window
        self.model = self._build_model()
        super().__init__(name, window=window)

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.window * 2,)))
        model.add(Dense(3))
        model.add(Activation('softmax'))

        memory = EpisodeParameterMemory(limit=1000, window_length=1)

        cem = CEMAgent(model=model, nb_actions=3, memory=memory,
                       batch_size=50, nb_steps_warmup=1000, train_interval=50, elite_frac=0.05)
        cem.compile()
        return cem

    def _load(self):
        self.model.load_weights(self.BASE_DIR + self.name + '/weights.h5f')

    def _train(self):
        os.mkdir(self.BASE_DIR + self.name)
        env = djq_utils.stock_env(self.model_name, self.etf_name, window=self.window, mode='all')
        self.model.fit(env, nb_steps=50000, visualize=False, verbose=0)
        self.model.save_weights(self.BASE_DIR + self.name + '/weights.h5f', overwrite=True)

    def get_action(self, observation):
        return self.model.forward(observation)


class DqnAgent(Agent):
    AGENT_NAME = 'DqnAgent'

    def __init__(self, name, window=5):
        self.window = window
        self.model = None
        super().__init__(name, window=window)

    def _build_model(self, args):
        assert type(args) == dict

        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.window * 2,)))
        for _ in range(args['n_layers']):
            model.add(Dense(args['layer_dense']))
            model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)

        policy = {'Boltzmann': BoltzmannQPolicy(),
                  'Eps_greedy': EpsGreedyQPolicy(eps=0.2),
                  'Eps_decay_greedy': LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                                           value_min=.1, value_test=.05,
                                                           nb_steps=args['episode'])}[args['policy']]

        dqn = {'DdqnAgent': DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=1000,
                                     enable_dueling_network=True, dueling_type='avg',
                                     target_model_update=1e-2, policy=policy),
               'DqnAgent': DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=1000,
                                    target_model_update=1e-2, policy=policy)}[self.AGENT_NAME]
        dqn.compile(Adam(lr=1e-3, decay=1e-6), metrics=['mae'])
        return dqn

    def _single_run(self, env, model, episode):
        model.fit(env, nb_steps=episode, visualize=False, verbose=0)
        his = model.test(env, nb_episodes=1, visualize=False)
        return his.history['episode_reward']

    def _param_run(self, env, model, episode):
        record = []
        pool = multiprocessing.Pool(processes=10)
        for _ in range(50):
            record.append(pool.apply_async(self._single_run, args=(env, model, episode)))
        pool.close()
        pool.join()
        env.mode = 'test' if env.mode == 'train' else 'train'
        res1 = np.average([r.get() for r in record])
        record = []
        pool = multiprocessing.Pool(processes=10)
        for _ in range(50):
            record.append(pool.apply_async(self._single_run, args=(env, model, episode)))
        pool.close()
        pool.join()
        res2 = np.average([r.get() for r in record])
        return res1 + res2

    def _load(self):
        with open(self.BASE_DIR + self.name + '/config.json', 'r') as f:
            args = json.load(f)
        self.model = self._build_model(args)
        self.model.load_weights(self.BASE_DIR + self.name + '/weights.h5f')

    def _train(self):
        os.mkdir(self.BASE_DIR + self.name)
        best_params = dict()
        best_profit = -float('inf')
        env = djq_utils.stock_env(self.model_name, self.etf_name, window=self.window, mode='train')
        for episode in [50000, 100000]:
            for policy in ['Boltzmann', 'Eps_greedy', 'Eps_decay_greedy']:
                for n_layers in range(3):
                    for layer_dense in [16, 32, 64]:
                        args = {'episode': episode, 'policy': policy, 'n_layers': n_layers, 'layer_dense': layer_dense}
                        model = self._build_model(args)
                        score = self._single_run(env, model, episode)
                        if score > best_profit:
                            best_params = args
        with open(self.BASE_DIR + self.name + '/config.json', 'w') as f:
            json.dump(best_params, f)
        env.mode = 'all'
        self.model = self._build_model(best_params)
        self.model.fit(env, nb_steps=best_params['episode'], visualize=False, verbose=0)
        self.model.save_weights(self.BASE_DIR + self.name + '/weights.h5f', overwrite=True)

    def get_action(self, observation):
        return self.model.forward(observation)


class DdqnAgent(DqnAgent):
    AGENT_NAME = 'DdqnAgent'


class MultiAgent(Agent):
    AGENT_NAME = 'MultiAgent'

    def __init__(self, name, subagents_list: list, window=5, agents_num=5):
        """
        :param name:str, consists of model name trained by djq_train_model, etf name correspondent to your model, id
        :param subagents_list: list, a list of agent classes which are preferred
        :param window:int, the length of the latest predict results and your positions
        :param agents_num: int or list of int, the agent number of each agent class
        """
        self.window = window
        if type(agents_num) == int:
            agents_num = [agents_num] * len(subagents_list)
        elif type(agents_num) != list:
            raise TypeError('agents_num should be int or list of int')
        if len(agents_num) != len(subagents_list):
            raise ValueError('length of subagents should equal length of agents_num')
        self.sub_agents = []
        super().__init__(name, window=window)
        for agentclass, agents_n in zip(subagents_list, agents_num):
            for i in range(agents_n):
                self.sub_agents.append(agentclass(self.model_name + '#' + self.etf_name + '#' + str(i+1),
                                                  window=self.window))

    def _load(self):
        pass

    def _train(self):
        os.mkdir(self.BASE_DIR + self.name)

    def get_action(self, observation):
        res = Counter()
        for sub_agent in self.sub_agents:
            res[sub_agent.get_action(observation)] += 1
        print(res)
        action = max(res)
        if res[1] == res[action]:
            action = 1
        return action


if __name__ == '__main__':
    agent = DqnAgent('SVM_target30_classify5_inx-399006_loss-r2_lda_2021#159915#1')