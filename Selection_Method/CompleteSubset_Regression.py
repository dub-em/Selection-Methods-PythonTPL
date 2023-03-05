import itertools
import random
import pandas as pd
from sklearn.metrics import mean_squared_error

class comletesubset_regression():
    """
    NAME
        comletesubset_regression

    DESCRIPTION
        Statistical Learning module for Python
        --------------------------------------

        comletesubset_regression is a Python module which implements 
        a statistical learning method for computing an aggregate prediction
        of a target variable using randomly sampled subsets of different 
        independent features in a given dataset, all having the same length.

    CLASS ARGUMENTS
        reg_model : regression model object
            A specific regression model object (linear, ridge, lasso, logic etc.)
    """

    def __init__(self, reg_model):
        self.reg_model = reg_model

    def __str__(self):
        return 'Best Subset Object using the {} method.'.format(self.reg_model)

    def aggregate_prediction(self, x_train, x_test, y_train, y_test, len_of_subset:int):
        """
        The aggregate_prediction() method takes in four sets of data (two for 
        testing and two for training), and a specified length for the subset to 
        be used. The itertools library is used to create a list of all possible 
        feature combinations of the specified length, and a sample of these subsets
        are taking to be used for the aggregate prediciton. The randomly sampled
        subsets are used to predict the target varibales separately and then their
        aggregate prediction is calculated. The mean square error between this 
        aggregate values and the actual values of the target variable is also computed.

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
        len_of_subset : int
            The length of subset of feature to use for aggregate prediction

        Returns
        -------
        y_predict
            A dataframe containing the separate prediction made using the various
            different sets of feature combinations (all having the same length). This
            dataframe also containg the total sum of the separate predictions, the
            aggregate prediction (vector mean of all the prediction), and the actual
            values of the target variable (for comparison purposes).
        pred_error
            The mean square error gotten from comparing the aggregate prediction and
            the actual values of the target variable.
        """

        #arguments are assigned to their respective variables.
        X_train, X_test = x_train, x_test
        Y_train, Y_test = y_train, y_test
        model = self.reg_model

        #creating a list of all possible combinations of the features, of a certain length.
        y_predict = {}
        var = list(X_train.columns)
        subsets = list(itertools.combinations(var, len_of_subset))

        #taking out a sample of subsets from the comlete set of all subsets, of a certain length.
        sampled_list = random.sample(subsets, int(len(subsets)*0.4))

        #using each of these subset of features to predict the target variable.
        for item in sampled_list:
            item_ = list(item)
            model.fit(X_train[item_], Y_train)
            y_predict[str(item)] = list(model.predict(X_test[item_]))
        y_predict = pd.DataFrame(y_predict)

        for item in sampled_list:
            y_predict[str(item)] = y_predict[str(item)].explode() 

        #computing the toatl sum of predictions and their aggregate, and comparing this the to target variable.
        y_predict['TotalSum'] = y_predict.sum(axis=1)
        y_predict['AggregatePred'] = y_predict['TotalSum']/len(sampled_list)
        y_predict['ActualTarg'] = Y_test
        y_predict.head(5)
        pred_error = mean_squared_error(y_predict['ActualTarg'], y_predict['AggregatePred'])       
        return (y_predict, pred_error)