import random

import numpy as np


class ActionTrainer(object):
    def __init__(self):
        self._eval_mode = True
        self.RANDOM_ACTION_P = 0.05

        self.random_default = 0.0
        self.random_bad = 0.8
        self.random_good = 0.1
        self.random_diff = 0.0

        self.range = (0.0, 0.0,)

        self.steps = [0, 0, 0, 0]

    def set_params(self, low, high, random_default, random_bad, random_good, random_diff):
        self.random_default = random_default
        self.random_bad = random_bad
        self.random_good = random_good
        self.random_diff = random_diff

        self.range = (low, high,)

    def get_ranges(self):
        return self.range

    def set_evaluate(self, mode):
        self._eval_mode = mode

    def reset_steps(self):
        self.steps = [0, 0, 0, 0]

    def choose_action(self, options):
        self.steps[0] += 1

        # ищем действие, которое обещает максимальную награду
        _l, _h = self.get_ranges()
        cl = sum((1 if expected_reward < _l else 0 for expected_reward, act in options))
        ch = sum((1 if expected_reward >= _h else 0 for expected_reward, act in options))

        random_p = self.random_default
        # все плохо
        if cl >= 7:
            random_p = self.random_bad
            self.steps[1] += 1

        # все хорошо
        if ch >= 7:
            random_p = self.random_good
            self.steps[2] += 1

        # есть разные варианты
        if cl >= 2 and ch >= 2:
            random_p = self.random_diff
            self.steps[3] += 1

        best_action = sorted(options, key=lambda x: x[0])[-1][1]
        # Иногда выбираем совершенно рандомное действие
        if (not self._eval_mode) and (random.random() < random_p):
            best_action = options[np.random.choice(len(options))][1]

        return best_action

    def choose_action_old(self, options):
        # ищем действие, которое обещает максимальную награду
        best_action = sorted(options, key=lambda x: x[0])[-1][1]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if (not self._eval_mode) and (random.random() < self.RANDOM_ACTION_P):
            best_action = options[np.random.choice(len(options))][1]

        return best_action
