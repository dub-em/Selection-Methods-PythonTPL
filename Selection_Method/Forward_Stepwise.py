class forward_stepwise():
    """
    NAME
        forward_stepwise

    DESCRIPTION
        Statistical Learning module for Python
        --------------------------------------

        forward_stepwise is a Python module which implements a statistical 
        learning method for selecting features (for predicting a target variable) 
        in a given dataset.

        This selection process is done by an iterative and eliminative 
        process of testing each feature and subsequently combining them
        in an incremental manner till the best set of features are found.

    CLASS ARGUMENTS
        reg_model : regression model object
            A specific regression model object (linear, ridge, lasso, logic etc.)
    """

    def __init__(self, reg_model):
        self.reg_model = reg_model

    def __str__(self):
        return 'Forward Stepwise Object using the {} method.'.format(self.reg_model)

    def select_features(self, x_train, x_test, y_train, y_test):
        """
        The select_features() method takes in four sets of data (two for 
        testing and two for training), and subjects these data to the
        forward stepwise method, so as to select the optimal features.

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
        final_list
            A list of the independent features selected by the algorithm.
        max(scores_)
            The predictive accuracy of the features in the final_list.
        """

        #arguments are assigned to their respective variables.
        X_train, X_test = x_train, x_test
        Y_train, Y_test = y_train, y_test
        model = self.reg_model

        #assigning initial variables for the algorithm.
        variable = list(X_train.columns)
        var = list(variable)
        final_list, scores_, prev_score = [], [], 0

        for i in range(len(variable)):
            temp_list = final_list

            #iterative assessment of each variable, while incrementally combing them.
            for item in var:
                temp_list = temp_list + [item]
                model.fit(X_train[temp_list], Y_train)
                scores_.append(model.score(X_test[temp_list], Y_test))
                temp_list = final_list

            index = scores_.index(max(scores_))
            val = var[index]
            score_change = max(scores_) - prev_score #checking different in previous and current best scores.

            #checking to see if the set criteria is broken, if there's an improvement in performance.
            if score_change > 0: 
                final_list.append(val)
                var.pop(index)
                prev_score = max(scores_)
                scores_ = []
            else:
                break        
        return (final_list, max(scores_))