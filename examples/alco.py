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

def load_data(src):
    # Load student alcohol consumption data
    df = pd.read_csv(src, sep=';')
    df.rename(columns={"G3": "target"}, inplace=True)
    df_small = df[['G1', 'G2', 'absences', 'Dalc', 'Walc', 'freetime', 'goout', 'target']]
    df_small.drop_duplicates(inplace=True)
    return df_small

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
        # Prepare dataset
        self.dataset = load_data(data_src)
        self.keys = sorted(self.dataset.columns.values)
        self.keys.remove('target')

        # Prepare a GP model to mimic the evaluation function
        self.gp = GaussianProcessRegressor(kernel=Matern(),
                                           n_restarts_optimizer=25,
                                           normalize_y=True)
        X = np.asarray(self.dataset[[*self.keys]].values)
        y = np.asarray(self.dataset['target'].values)

        ur = unique_rows(X)

        self.gp.fit(X[ur], y[ur])

    def eval(self, **kw_args):
        x = []
        for key in self.keys:
            x.append(kw_args[key])

        miu = self.gp.predict(np.asarray(x).reshape(1, -1))[0]
        return miu

    def run(self):
        # Create a BO object
        bo = BayesianOptimization(self.eval,
                                  {'G1': (0, 20),
                                   'G2': (0, 20),
                                   'absences': (0, 100),
                                   'Dalc': (0, 5),
                                   'Walc': (0, 5),
                                   'freetime': (1, 5),
                                   'goout': (1, 5)})

        # Get the input data without output
        # data = self.dataset.drop(['target'], axis=1).to_dict('list')

        # Once we are satisfied with the initialization conditions
        # we let the algorithm do its magic by calling the maximize()
        # method.
        # self, dataset, init_points=5, n_iter=25, kappa=2.576, xi=0.0, eta=1.01, **gp_params):
        bo.maximize_mixed(self.dataset, init_points=5, n_iter=15, kappa=3.29, eta=0.0005)

        # The output values can be accessed with self.res
        print(bo.res['max'])
        print(bo.res['all'])

if __name__ == "__main__":
    print(__doc__)
    App(sys.argv[1]).run()

