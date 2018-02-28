import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    def __init__(self, name="noname", history_data=int(50000)):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        self.name = name
        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = 5  # выберите число лучей ладара; например, 5
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = Network([self.rays + 4,
                                   # self.rays + 4,
                                   # внутренние слои сети: выберите, сколько и в каком соотношении вам нужно
                                   # например, (self.rays + 4) * 2 или просто число
                                   #(self.rays + 4) * 2,
                                   6,
                                   1],
                                  output_function=lambda x: x, output_derivative=lambda x: 1)

        # Нормировочный слой
        # norm_weights = np.diag([2.0, 1.0, 0.4] + [0.2] * (self.rays - 2) + [0.4, 1.0, 1.0])
        # norm_biases = -1.0 * np.array([2.0, 0.0, 0.5] + [0.7] * (self.rays - 2) + [0.5, 0.0, 0.0]).reshape(-1, 1)
        # self.neural_net.weights[0] = norm_weights
        # self.neural_net.biases[0] = norm_biases

        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0
        self.RANDOM_ACTION_P = 0.30

    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без него не пройдёт нормально eval
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def clear_history(self):
        self.sensor_data_history.clear()
        self.chosen_actions_history.clear()
        self.reward_history.clear()

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = []
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
            for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map.append((predicted_reward, action))

        # ищем действие, которое обещает максимальную награду
        best_action = sorted(rewards_to_controls_map, key=lambda x: x[0])[-1][1]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if (not self.evaluate_mode) and (random.random() < self.RANDOM_ACTION_P):
            best_action = rewards_to_controls_map[np.random.choice(len(rewards_to_controls_map))][1]

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)

        return best_action

    def receive_feedback(self, reward, train_every=50, reward_depth=7):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        # действие, но и предшествующие)
        i = -1
        self.reward_history.append(reward)


