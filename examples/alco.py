import sys
sys.path.append("./")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
"""
Perform Bayesian Optimization on Student Alcohol Consumption dataset

Usage
-----
python alco.py [<student_data.csv>]

Output
------
"""

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]

class App(object):
    def __init__(self, data_src):
        self.data_src = data_src

        # Prepare dataset placeholders
        self.dataset = None
        self.keys = None

        # Prepare a GP model to mimic the evaluation function
        self.gp = GaussianProcessRegressor(kernel=Matern(),
                                           n_restarts_optimizer=25,
                                           normalize_y=True)

    def eval(self, **kw_args):
        x = []
        for key in self.keys:
            x.append(kw_args[key])

        miu = self.gp.predict(np.asarray(x).reshape(1, -1))[0]
        return miu
    
    def load_data(self, attrs):
        # Load student alcohol consumption data
        df = pd.read_csv(self.data_src, sep=';')
        df.rename(columns={"G3": "target"}, inplace=True)
        attrs += ['target']
        df_small = df[attrs]
        df_small.drop_duplicates(inplace=True)

        # Prepare dataset
        self.dataset = df_small
        self.keys = sorted(self.dataset.columns.values)
        self.keys.remove('target')

    def initialize_objective(self):
        """ Initialize objective function using a GP model. """

        X = np.asarray(self.dataset[[*self.keys]].values)
        y = np.asarray(self.dataset['target'].values)

        ur = unique_rows(X)

        self.gp.fit(X[ur], y[ur])

    def run(self):
        # Bounds for chosen attributes
        pbounds = {'G1': (0, 20),
                   'G2': (0, 20),
                   'absences': (0, 93),
                   'age': (15, 22),
                   'Dalc': (0, 5),
                   'Walc': (0, 5)}

        # Create a BO object
        bo_mixed = BayesianOptimization(self.eval, pbounds)

        # Read data with the chosen attributes
        self.load_data(list(pbounds.keys()))
        self.initialize_objective()

        # Once we are satisfied with the initialization conditions
        # we let the algorithm do its magic by calling the maximize()
        # method.
        # self, dataset=None, init_points=5, n_iter=25, acq="mixed", kappa=2.576, xi=0.0, eta=1.01, **gp_params
        bo_mixed.maximize_mixed(self.dataset, init_points=5, n_iter=5, kappa=3.29, eta=0.0005)

        # The output values can be accessed with self.res
        print(bo_mixed.res['max'])
        print(bo_mixed.res['all'])

if __name__ == "__main__":
    print(__doc__)
    App(sys.argv[1]).run()

