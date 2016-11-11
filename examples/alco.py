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
        self.X = None
        self.y = None

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
        full_attrs = attrs + ['target']
        print("Full", full_attrs)
        print("Attr", attrs)
        df_selected = df[full_attrs]
        df_selected.drop_duplicates(inplace=True)

        # Prepare dataset
        self.dataset = df_selected[attrs]
        self.keys = sorted(self.dataset.columns.values)

        # Save data for GP model
        self.X = np.asarray(df_selected[[*self.keys]].values)
        self.y = np.asarray(df_selected['target'].values)

    def initialize_objective(self):
        """ Initialize objective function using a GP model. """

        ur = unique_rows(self.X)

        self.gp.fit(self.X[ur], self.y[ur])

    def run(self):
        # Bounds for chosen attributes
        pbounds = {'G1': (0, 20),
                   'G2': (0, 20),
                   'absences': (0, 93),
                   'age': (15, 22),
                   'Dalc': (0, 5),
                   'Walc': (0, 5)}

        # Read data with the chosen attributes
        self.load_data(list(pbounds.keys()))
        self.initialize_objective()

        # Create a BO object for mixed strategy
        print("--- Evaluating Bayesian Optimization using Mixed strategy ---")
        bo_mixed = BayesianOptimization(self.eval, pbounds)

        # Once we are satisfied with the initialization conditions
        # we let the algorithm do its magic by calling the maximize()
        # method.
        bo_mixed.maximize_mixed(self.dataset, init_points=5, n_iter=5, kappa=3.29, eta=0.0005)

        # The output values can be accessed with self.res
        print(bo_mixed.res['max'])
        print(bo_mixed.res['all'])

        # BO object for Expected Improvement
        print("--- Evaluating Bayesian Optimization using EI ---")
        bo_ei = BayesianOptimization(self.eval, pbounds)
        bo_ei.maximize_mixed(self.dataset, init_points=5, n_iter=5, acq="ei")
        print(bo_ei.res['max'])
        print(bo_ei.res['all'])

        # BO object for Probability of Improvement
        print("--- Evaluating Bayesian Optimization using POI ---")
        bo_poi = BayesianOptimization(self.eval, pbounds)
        bo_poi.maximize_mixed(self.dataset, init_points=5, n_iter=5, acq="poi")
        print(bo_poi.res['max'])
        print(bo_poi.res['all'])

        # BO object for Upper Confidence Bound
        print("--- Evaluating Bayesian Optimization using POI ---")
        bo_ucb = BayesianOptimization(self.eval, pbounds)
        bo_ucb.maximize_mixed(self.dataset, init_points=5, n_iter=5, acq="ucb", kappa=3.29)
        print(bo_ucb.res['max'])
        print(bo_ucb.res['all'])


if __name__ == "__main__":
    print(__doc__)
    App(sys.argv[1]).run()

