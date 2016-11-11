import sys
import pickle
sys.path.append("./")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
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
    def __init__(self, data, train=5, iterations=10, kappa=3.29, eta=0.0005):
        # Save arguments
        self.data_src = data
        self.train = train
        self.iterations = iterations
        self.kappa = kappa
        self.eta = eta

        print(self.kappa)

        # Prepare dataset placeholders
        self.dataframe = None
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
        """
        Load the data from given source. Only choose attributes that are
        given the bounds.

        Parameters
        ----------
        :param attrs:
            The list of attributes/columns to pick from.
        """
        # Load student alcohol consumption data
        df = pd.read_csv(self.data_src, sep=';')
        df.rename(columns={"G3": "target"}, inplace=True)
        full_attrs = attrs + ['target']
        df_selected = df[full_attrs]
        df_selected.drop_duplicates(inplace=True)

        # Prepare dataset
        self.dataframe = df_selected
        self.dataset = df_selected[attrs]
        self.keys = sorted(self.dataset.columns.values)

        # Save data for GP model
        self.X = np.asarray(df_selected[[*self.keys]].values)
        self.y = np.asarray(df_selected['target'].values)

    def initialize_objective(self):
        """ Initialize objective function using a GP model. """

        ur = unique_rows(self.X)

        self.gp.fit(self.X[ur], self.y[ur])

    def plot_bos(self, profiles):
        """
        Plot the graph for the means over number of iterations

        Parameters
        ----------
        :param profiles:
            An dict of data points for each of the different strategy. 
        """

        # Initialize axis
        i = np.arange(self.iterations)

        values = profiles.values()
        plt.figure()
        plt.plot(i, values[0], 'r--', i, values[1], 'bs', i, values[2], 'g^')
        plt.show()

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

        # Randomly choose some training data
        points_df = self.dataframe.sample(self.train)
        print(points_df)

        # Create a BO object for mixed strategy
        print("--- Evaluating Bayesian Optimization using Mixed strategy ---")
        bo_mixed = BayesianOptimization(self.eval, pbounds)
        bo_mixed.initialize_df(points_df)

        # Once we are satisfied with the initialization conditions
        # we let the algorithm do its magic by calling the maximize()
        # method.
        bo_mixed.maximize_mixed(self.dataset, init_points=self.train, n_iter=self.iterations,
                                kappa=self.kappa, eta=self.eta)

        # The output values can be accessed with self.res
        print(bo_mixed.res['max'])
        print(bo_mixed.res['all'])

        # BO object for Expected Improvement
        print("--- Evaluating Bayesian Optimization using EI ---")
        bo_ei = BayesianOptimization(self.eval, pbounds)
        bo_ei.initialize_df(points_df)
        bo_ei.maximize_mixed(self.dataset, init_points=self.train, n_iter=self.iterations, acq="ei")
        print(bo_ei.res['max'])
        print(bo_ei.res['all'])

        # BO object for Probability of Improvement
        print("--- Evaluating Bayesian Optimization using POI ---")
        bo_poi = BayesianOptimization(self.eval, pbounds)
        bo_poi.initialize_df(points_df)
        bo_poi.maximize_mixed(self.dataset, init_points=self.train, n_iter=self.iterations, acq="poi")
        print(bo_poi.res['max'])
        print(bo_poi.res['all'])

        # BO object for Upper Confidence Bound
        print("--- Evaluating Bayesian Optimization using POI ---")
        bo_ucb = BayesianOptimization(self.eval, pbounds)
        bo_ucb.initialize_df(points_df)
        bo_ucb.maximize_mixed(self.dataset, init_points=self.train, n_iter=self.iterations, acq="ucb", kappa=self.kappa)
        print(bo_ucb.res['max'])
        print(bo_ucb.res['all'])

        # Combine the results into profiles
        # profiles = {"mixed": bo_mixed.res['all']['values'],
        #             "ei": bo_ei.res['all']['values'],
        #             "poi": bo_poi.res['all']['values'],
        #             "ucb": bo_ucb.res['all']['values']}

        profiles_full = {"mixed": bo_mixed.res,
                         "ei": bo_ei.res,
                         "poi": bo_poi.res,
                         "ucb": bo_ucb.res}

        pickle.dump(profiles_full, open("profiles.p", "wb"))
        print("Done pickling")
        # self.plot_bos(profiles)

if __name__ == "__main__":
    print(__doc__)

    # Get program arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", type=str, help="str: path to data file", required=True)
    ap.add_argument("-i", "--iterations", type=int, help="int: number of iterations", default=10)
    ap.add_argument("-e", "--eta", type=float, help="float: learning rate for Hedge algorithm", default=0.005)
    ap.add_argument("-k", "--kappa", type=float, help="float: exploration/exploitation trade off for GP-UCB", default=3.29)
    ap.add_argument("-t", "--train", type=int, help="int: number of training data", default=5)
    args = vars(ap.parse_args())

    App(**args).run()