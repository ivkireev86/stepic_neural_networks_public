# from HW_3.cars import *
import argparse
import random
from functools import reduce

import numpy as np
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map
from cars.world import SimpleCarWorld
from tqdm import tqdm_notebook


def create_map(seed, agent):
    np.random.seed(seed)
    random.seed(seed)
    m = generate_map(8, 5, 3, 3)
    w = SimpleCarWorld([agent], m, SimplePhysics, None, timedelta=0.2)
    return w


def norm_y(y):
    return y / 1.0


def mine_data(log_list, dyn_plot, agent, action_trainer_params,
              clear_history, map_seed, steps,
              epochs, mini_batch_size, eta, reward_shift, alpha):
    agent.action_trainer.set_params(**action_trainer_params)
    if clear_history:
        agent.clear_history()

    # сбрасываем статистику
    agent.action_trainer.reset_steps()

    # mine data
    w = create_map(seed=map_seed, agent=agent)
    w.run(steps=tqdm_notebook(range(steps), desc="train", leave=False), visual=False, save=False)

    steps_str = "total: {:d}, bad: {:.4f}, good: {:.4f}, diff: {:.4f}, other: {:.4f}".format(
        agent.action_trainer.steps[0],
        agent.action_trainer.steps[1] / agent.action_trainer.steps[0],
        agent.action_trainer.steps[2] / agent.action_trainer.steps[0],
        agent.action_trainer.steps[3] / agent.action_trainer.steps[0],
        1 - sum(agent.action_trainer.steps[1:]) / agent.action_trainer.steps[0])

    # prepare train data
    X_train = np.concatenate([agent.sensor_data_history, agent.chosen_actions_history], axis=1)
    y_train = np.array(agent.reward_history)
    # сглаживаем пики, чтобы сеть небольшая сеть могла дать адекватное предсказание
    mean_train_revard = y_train.mean()
    y_train = norm_y(y_train)

    y_train = np.pad(y_train, (0, 2 * reward_shift), mode="constant")
    y_train = reduce(lambda a, b: a + b, [y_train[i:(-2 * reward_shift + i)] * (alpha ** i)
                                          for i in range(reward_shift)])

    # train NN
    train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
    train_rmse_before = agent.neural_net.evaluate(X_train.T, y_train)
    next(agent.neural_net.SGD(training_data=train_data,
                              epochs=tqdm_notebook(range(epochs), desc="SGD", leave=False),
                              mini_batch_size=mini_batch_size,
                              eta=eta))
    train_rmse_after = agent.neural_net.evaluate(X_train.T, y_train)

    collisions = sum((1 if x < 0 and 0 < len(agent.reward_history) - i < steps else 0
                      for i, x in enumerate(agent.reward_history))) / steps

    # evaluate
    w = create_map(seed=map_seed, agent=agent)
    mean_test_reward = w.evaluate_agent(agent, steps=tqdm_notebook(range(1200), desc="test", leave=False), visual=False)
    circles = w.circles[agent]

    log_message = f"""map_seed = {map_seed}
train_rmse_before = {train_rmse_before:.9f}, train_rmse_after = {train_rmse_after:.9f}, 
mean_train_revard = {mean_train_revard:.3f}, mean_test_reward  = {mean_test_reward:.3f},
steps = {steps_str},
collisions = {collisions:3f}, circles = {circles:2f}"""

    log_list.append(log_message)

    if not dyn_plot:
        chart_count = agent.neural_net.num_layers
        plt.figure(figsize=(5 * chart_count, 2))
        for l in range(chart_count - 1):
            plt.subplot(1, chart_count, l + 1)
            ax = sns.heatmap(agent.neural_net.weights[l])

        plt.subplot(1, chart_count, chart_count)
        plt.text(0.05, 0.95, log_message, size=12, ha='left', va='top', family='monospace')

        plt.show()
    else:
        print(log_message)

    return circles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--steps", type=int)
    parser.add_argument("-f", "--filename", type=str)
    parser.add_argument("-e", "--evaluate", type=bool)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    print(args)

    steps = args.steps
    seed = args.seed if args.seed else 23
    np.random.seed(seed)
    random.seed(seed)
    m = generate_map(8, 5, 3, 3)

    if args.filename:
        agent = SimpleCarAgent.from_file(args.filename)
        w = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
        if args.evaluate:
            print(w.evaluate_agent(agent, steps))
        else:
            w.set_agents([agent])
            w.run(steps)
    else:
        SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2).run(steps)
