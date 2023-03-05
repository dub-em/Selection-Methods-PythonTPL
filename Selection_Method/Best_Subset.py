import itertools
import numpy as np
import pandas as pd

class best_subset():
    """
    NAME
        best_subset

    DESCRIPTION
        Statistical Learning module for Python
        --------------------------------------

        best_subset is a Python module which implements a statistical 
        learning method for selecting the best feature set (for 
        predicting a target variable) in a given dataset.

        This selection process is creating a set of all possible set
        of combination of a given set of independent features in a dataset.
        The specified model is then traied on all these sets and the 
        best subset of features is selected.

    CLASS ARGUMENTS
        reg_model : regression model object
            A specific regression model object (linear, ridge, lasso, logic etc.)
    """

    def __init__(self, reg_model):
        self.reg_model = reg_model

    def __str__(self):
        return 'Best Subset Object using the {} method.'.format(self.reg_model)

    def select_features(self, x_train, x_test, y_train, y_test):
        """
        The select_features() method takes in four sets of data (two for 
        testing and two for training), and subjects these data to the
        best subset method, so as to select the optimal features.

        Parameters
        ----------
        x_train : pandas.core.frame.DataFrame
            The dataset containing the independent variables used for training.
        X_test : pandas.core.frame.DataFrame
            The dataset containing the independent variables used for testing.
        y_train : pandas.core.frame.DataFrame
            The dataset containing the target variable used for training.
        y_test : pandas.core.frame.DataFrame
            The dataset containing the target variable used for testing.

        Returns
        -------
        bestsubset
            A list of the best performing independent features.
        max_score
            The predictive accuracy of best performing independent features.
        """

        #arguments are assigned to their respective variables.
        X_train, X_test = x_train, x_test
        Y_train, Y_test = y_train, y_test
        model = self.reg_model

        #creating a list of all possible combinations of the columns in the dataset
        var = list(X_train.columns)
        set_list = []
        for i in range(1,len(var)+1):
            sets = list(itertools.combinations(var, i))
            for item in sets:
                set_list.append(list(item))

        #computing the model score of each combination and storing in a list
        set_score = []
        for item in set_list:
            model.fit(X_train[item], Y_train)
            set_score.append(model.score(X_test[item], Y_test))
        
        #extracting the best score and its corresponding subset of columns
        max_score = max(set_score)
        max_index = set_score.index(max_score)
        bestsubset = set_list[max_index]       
        return (bestsubset, max_score)