import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class bootstrap():
    """
    NAME
        bootstrap

    DESCRIPTION
        Statistical Learning module for Python
        --------------------------------------

        bootstrap is a Python module which implements a statistical learning method
        for checking how well a dataset represents the data source.

        This process starts with random sampling and replacing of datapoints in the
        dataset. The parameter of interest (mean, standard deviation, model error etc.)
        is then calculated for the new dataset. This process is repeated a specified 
        number of times, and the same parameter is calculated and stored. After this 
        process is concluded, the parameter calculated for the original dataset is 
        checked (using hyptohesis test) against the set of parameters calculated for all
        the sampled datasets to see how well the original dataset represents the data
        source.

    CLASS ARGUMENTS
        reg_model : regression model object
            A specific regression model object (linear, ridge, lasso, logic etc.)
        non_model_param : regular python object or function
            A custom function or python object written for a specific function
            (e.g. mean, standard deviation, variance etc.)
    """

    def __init__(self, reg_model=None, non_model_param=None):
        self.reg_model = reg_model
        self.param = non_model_param

    def __str__(self):
        return 'Bootstrap Object.'

    def modelerror_check(self, x, y, iteration_num:int):
        """
        The modelerror_check() method takes two sets of data (one for the
        independent variables and another for the target variable).
        The last parameter is the number of iterations for the sampling
        process, which has its mandatory datatype set to integer.

        Parameters
        ----------
        x : pandas.core.frame.DataFrame
            The dataset containing the independent variables.
        y : pandas.core.frame.DataFrame
            The dataset containing the target variable.
        iteration_num: integer
            The number of iterations for the sampling process.

        Returns
        -------
        boot_err_set
            A pandas dataframe containing the model errors gotten from the sampled dataset.
        z_value
            The z_value gotten from the hypothesis test, comparing the model error of the
            original dataset to the model error gotten from the sampled datasets.
        """

        if self.reg_model == None:
            raise Exception("Sorry, you can't call up this method without specifying a regression model object.")
        else:
            #arguments are assigned to their respective variables.
            X = x
            Y = y
            model = self.reg_model

            #computing the error for the original dataset for hypothesis test later.
            model.fit(X, Y)
            y_pred = model.predict(X)
            samp_err = mean_squared_error(Y, y_pred)

            #sampling the dataset and calculating the parameter (model error) for each sample.
            boot_dict = {'Error':[]}

            boot_num = iteration_num
            for i in range(boot_num):
                boot = np.random.choice(list(range(X.shape[0])), replace=True, size=X.shape[0])
                boot_sam = X.iloc[boot,:]
                boot_tar = Y.iloc[boot,:]
                model.fit(boot_sam, boot_tar)
                y_pred = model.predict(boot_sam)
                boot_err = mean_squared_error(boot_tar, y_pred)
                boot_dict['Error'].append(boot_err)
            boot_err_set = pd.DataFrame(boot_dict)

            #hypothesis test
            m = boot_err_set['Error'].mean()
            f = samp_err
            std = boot_err_set['Error'].std()
            n = np.sqrt(boot_err_set.size)
            z_value = (f - m)/(std/n)
            return (boot_err_set, z_value)


    def param_check(self, x, iteration_num:int):
        """
        The param_check() method takes  one set of data which the 
        specified non-model parameter object will be executed on.
        The last parameter is the number of iterations for the sampling
        process, which has its mandatory datatype set to integer.

        Parameters
        ----------
        x : pandas.core.frame.DataFrame
        iteration_num: integer
            The number of iterations for the sampling process.

        Returns
        -------
        boot_param_set
            A pandas dataframe containing the calculated parameters gotten from the sampled dataset.
        z_value
            The z_value gotten from the hypothesis test, comparing the parameter of the
            original dataset to the model error gotten from the sampled datasets.
        """

        if self.param == None:
            raise Exception("Sorry, you can't call up this method without specifying a custom function for the parameter you want to calculate.")
        else:
            #arguments are assigned to their respective variables.
            X = pd.DataFrame(x)
            func = self.param

            #computing the parameter for the original dataset for hypothesis test later.
            samp_param = func(X.values)

            #sampling the dataset and calculating the parameter (custom function) for each sample.
            boot_dict = {'Param':[]}

            boot_num = iteration_num
            for i in range(boot_num):
                boot = np.random.choice(list(range(X.shape[0])), replace=True, size=X.shape[0])
                boot_sam = X.iloc[boot,:]
                boot_dict['Param'].append(func(boot_sam.values))
            boot_param_set = pd.DataFrame(boot_dict)

            #hypothesis test
            m = boot_param_set['Param'].mean()
            f = samp_param
            std = boot_param_set['Param'].std()
            n = np.sqrt(boot_param_set.size)
            z_value = (f - m)/(std/n)
            return (boot_param_set, z_value)


