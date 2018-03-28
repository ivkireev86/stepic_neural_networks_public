"""
Меняем сетку на лес
"""

import random
import numpy as np
from collections import Iterable
from sklearn.ensemble import RandomForestRegressor

class RandomForestWrapper(object):
    def __init__(self, **kwargs):
        self.reg = RandomForestRegressor(**kwargs)
        self.fitted = False

    def __repr__(self):
        raise NotImplementedError

    def feedforward(self, a):
        if self.fitted:
            return self.reg.predict(a.T)
        else:
            return np.zeros(a.shape[1])

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        if isinstance(epochs, Iterable):
            pass
        elif type(epochs) is int:
            epochs = range(epochs)
        else:
            RuntimeError("steps должен быть числом, итератором или None ")

        _first = True
        for _ in epochs:
            if _first:
                X_train = np.array([x[0].flatten() for x in training_data])
                Y_train = np.array([x[1] for x in training_data])

                self.reg.fit(X_train, Y_train)
                self.fitted = True
                _first = False
            else:
                pass

        if test_data is None:
            yield None

    def update_mini_batch(self, mini_batch, eta):
        raise NotImplementedError

    def backprop(self, x, y):
        raise NotImplementedError

    def evaluate(self, x_test, y_test):
        """
        Вернуть количество тестовых примеров, для которых нейронная сеть
        возвращает правильный ответ. Обратите внимание: подразумевается,
        что выход нейронной сети - это индекс, указывающий, какой из нейронов
        последнего слоя имеет наибольшую активацию.
        """
        y_prediction = self.feedforward(x_test)
        return (((y_test - y_prediction) ** 2).sum() / y_test.size) ** 0.5
