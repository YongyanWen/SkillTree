import os

import numpy as np
from sklearn.datasets import make_classification
from strl.components.params import get_args
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import random
import h5py
import pickle

# logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

class tree_classifier():
    def __init__(self, max_depth=5):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, criterion='gini')

    def train(self, X, Y):
        self.clf.fit(X, Y)

class tree_regressor():
    def __init__(self, max_depth=5):
        self.clf = DecisionTreeRegressor(max_depth=max_depth, criterion='squared_error')

    def train(self, X, Y):
        self.clf.fit(X, Y)


if __name__ == '__main__':
    args = get_args()
    seed_value = 1000
    random.seed(seed_value)
    np.random.seed(seed_value)

    dataset_size = 1000
    env = 'kitchen_mkbl'

    file_path = os.path.join(args.path, f'1000_{dataset_size}.h5')
    file = h5py.File(file_path, 'r')

    dataset = file['traj']
    state = dataset['states'][:]
    hl_action_index = dataset['hl_action_index'][:]

    max_depth = 6

    tree = tree_classifier(max_depth=max_depth)

    tree.train(state, hl_action_index)

    print(f'leaf: {tree.clf.get_n_leaves()}')
    print(f'depth: {tree.clf.get_depth()}')
    print(f'node_count: {tree.clf.tree_.node_count}')
    print(f'importance: {tree.clf.feature_importances_}')
    # print(f'gini: {tree.clf.tree_.impurity}')
    print(f'test_score: {tree.clf.score(state, hl_action_index)}')
    # print(f'test_score: {tree.clf.score(state, action)}')

    model_name = os.path.join(args.path, f'cart_fine_{dataset_size}_d{max_depth}.pkl')

    with open(model_name, 'wb') as f:
        print(f'write to {model_name}')
        pickle.dump(tree.clf, f)
