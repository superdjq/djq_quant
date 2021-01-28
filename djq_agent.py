import djq_utils
from abc import ABCMeta,abstractmethod
import os, json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, MaxBoltzmannQPolicy, LinearAnnealedPolicy
from rl.policy import GreedyQPolicy, EpsGreedyQPolicy, BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory
from rl.agents import CEMAgent
from collections import Counter


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


class DqnAgent(Agent):
    AGENT_NAME = 'DqnAgent'

    def __init__(self, name, window=5):
        self.window = window
        self.model = self._build_model()
        super().__init__(name, window=window)

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.window * 2,)))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)

        policy = BoltzmannQPolicy()

        dqn = DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=1000,
                       target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3, decay=1e-6), metrics=['mae'])
        return dqn

    def _load(self):
        self.model.load_weights(self.BASE_DIR + self.name + '/weights.h5f')

    def _train(self):
        os.mkdir(self.BASE_DIR + self.name)
        env = djq_utils.stock_env(self.model_name, self.etf_name, window=self.window, mode='all')
        self.model.fit(env, nb_steps=50000, visualize=False, verbose=0)
        self.model.save_weights(self.BASE_DIR + self.name + '/weights.h5f', overwrite=True)

    def get_action(self, observation):
        return self.model.forward(observation)


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


class DdqnAgent(Agent):
    AGENT_NAME = 'DdqnAgent'

    def __init__(self, name, window=5):
        self.window = window
        self.model = self._build_model()
        super().__init__(name, window=window)

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + (self.window * 2,)))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.add(Activation('linear'))

        memory = SequentialMemory(limit=50000, window_length=1)

        policy = BoltzmannQPolicy()

        dqn = DQNAgent(model=model, nb_actions=3, memory=memory, nb_steps_warmup=1000,
                       enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
        dqn.compile(Adam(lr=1e-3, decay=1e-6), metrics=['mae'])
        return dqn

    def _load(self):
        self.model.load_weights(self.BASE_DIR + self.name + '/weights.h5f')

    def _train(self):
        os.mkdir(self.BASE_DIR + self.name)
        env = djq_utils.stock_env(self.model_name, self.etf_name, window=self.window, mode='all')
        self.model.fit(env, nb_steps=50000, visualize=False, verbose=0)
        self.model.save_weights(self.BASE_DIR + self.name + '/weights.h5f', overwrite=True)

    def get_action(self, observation):
        return self.model.forward(observation)


class MultiAgent(Agent):
    AGENT_NAME = 'MultiAgent'

    def __init__(self, name, subagents_list: list, window=5, agents_num=5):
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
                self.sub_agents.append(agentclass(self.model_name + '#' + self.etf_name + '#' + str(i+1), window=self.window))

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


