# Stepwise Selection

Select the optimal features in a dataset using the stepwise method.

## Instructions

1. Install:

pip install model_selection

2. Plug in your train and test dataset, and your preferred algorithm.

# Forward_Stepwise
from model_selection import Forward_Stepwise

# initialize forward_stepwise object, inputting your already split train and test dataframes, and your already created regression model object.

selection = Forward_Stepwise.forward_stepwise(x_train, x_test, y_train, y_test, linear_model)

# select the best features using the stepwise algorithm through the .select() method.

final_list, final_score = selection.select()

# Backward_Stepwise
from model_selection import Backward_Stepwise

# initialize backward_stepwise object, inputting your already split train and test dataframes, and your already created regression model object.

selection = Backward_Stepwise.backard_stepwise(x_train, x_test, y_train, y_test, linear_model)

# select the best features using the stepwise algorithm through the .select() method.

final_list, final_score = selection.select()