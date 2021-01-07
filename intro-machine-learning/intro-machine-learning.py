# Course: https://www.kaggle.com/learn/intro-to-machine-learning
# Recommended IDE: https://www.spyder-ide.org/  

# Pandas is the primary tool for exploring and manipulating data.
import pandas as pd

melb_h_data_file_path = 'data/melb_data.csv'
# Read the comma-separated values (csv) file into a data structure called DataFrame.
melb_h_df = pd.read_csv(melb_h_data_file_path)

# Drop the rows where at least one element is missing.
# Empty brackets will make 'axis' argument default to '0'.
melb_h_df = melb_h_df.dropna() 

print(melb_h_df.describe())

# Pull out a variable with dot-notation. This single column is stored in a Series, which is like a DataFrameRead a comma-separated values (csv) file into DataFrame. with
# only a single column of data. By convention, the prediction target is called 'y'.
y = melb_h_df.Price

print(y.describe())

# Columns inputted into a model (and later used to make predictions) are called "features".
# Select multiple features by providing a list of column names. By convention, this data is called X.
print(melb_h_df.columns)
melb_h_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_h_df[melb_h_features]

print(X.describe())
print(X.head())

# Scikit-learn is popular for modeling data stored in DataFrames.
# Key steps to building a model are;
# Define 
# Fit - Capture patterns from provided data
# Predict
# Evaluate
#
# Deploy

# Define decision tree regression model with scikit-learn.
# For info about this model type: https://scikit-learn.org/stable/modules/tree.html#tree
from sklearn.tree import DecisionTreeRegressor
melb_h_model = DecisionTreeRegressor(random_state=1)

# Fit model
melb_h_model.fit(X, y)

# Predict house prices using newy fitted model.
# Training data can evaluate the model because prices are already known.
print("The predictions are:")
predicted_melb_h_prices = melb_h_model.predict(X)
print(predicted_melb_h_prices)

# Mean Absolute Error (MAE) is one metric to evaluate model quality,
# i.e. compare predicted to actual values.
print("MAE of the model (difference between predicted dollar values to actual):")
from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y, predicted_melb_h_prices)) 

# Above MAE is an "in-sample" score, i.e. used a single sample of data to both
# build and evaluate the model. This is bad practice because inaccurate 
# prediction patterns might be found which only hold true for the training data itself. 
# E.g. houses with green doors might be coincidently more expensive in the training data, 
# but not in reality, so predicting house price based on door color is inaccurate. 
#
# Instead of evaluating with the same training data, use a different sample 
# called "validation" data.

# Split data into training and validation data for both features (X) and target (y).
from sklearn.model_selection import train_test_split
train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=0)
 
# Define another decision tree model.
melb_h_model2 = DecisionTreeRegressor(random_state=1)

# Fit model using training data; subset of overall data.
melb_h_model2.fit(train_X, train_y)

# Predict house prices using the validation data sample.
print("The predictions are:")
validation_predictions = melb_h_model2.predict(validation_X)
print(validation_predictions)
print("MAE of the model ($):")
print(mean_absolute_error(validation_y, validation_predictions))

# Above MAE is an "out-of-sample" score, so it's probably a bigger error.
# Since the model accuracy can be measured, it's possible to improve it by experimenting
# with different features or even different model types.

# DecisionTreeRegressor can take many arguments, the most important ones determine the tree's depth,
# e.g. max_leaf_nodes which has a default of None. A tree's depth is a measure of how many 
# splits it makes before coming to a prediction. 

# Overfitting is where a model matches the training data almost perfectly,
# but does poorly in validation and other new data. It can happen when a tree
# is too deep so that the dataset is divided out into too many groups. These groups,
# i.e. leaves of the tree, would have too few data which leads to predicitions that
# are close to that data but unreliable for new data.
#
# Controlling the tree depth is key to avoid overfitting, but if the tree is too shallow
# then each leaf would not divide the dataset into distinct groups. This makes it hard
# for the model to find important patterns and distinctions in the data in order to 
# make accurate preditions. This is underfitting.
#
# The right balance must be found between under and over fitting the model.

# Function to get MAE of a decision tree, for the given data and max leaf nodes. 
# This helps to compare MAE scores from different values for max_leaf_nodes.
def get_dt_mae(max_leaf_nodes, train_X, validation_X, train_y, validation_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    predicted_value = model.predict(validation_X)
    mae = mean_absolute_error(validation_y, predicted_value)
    return mae
 
# Determine the optimal max_leaf_nodes value from a given range of candidate values.
# The optimal one gives the lowest MAE score.  
# All MAE scores are calculated and mapped to their corresponding max_leaf_nodes value
# using a dictionary data structure, from which the lowest MAE is selected.
# See: https://docs.python.org/3/library/stdtypes.html#dict
#
# Lowest MAE is updated each time a new lowest value is calculated, and the corresponding
# max_leaf_nodes is considered the new optimal. 
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]  
optimal_max_leaf_nodes = None
lowest_mae = None
for max_leaf_nodes in candidate_max_leaf_nodes:
    current_mae = get_dt_mae(max_leaf_nodes, train_X, validation_X, train_y, validation_y)
    print(f"Max leaf nodes: {max_leaf_nodes}. Mean Absolute Error: {current_mae:,.0f}")    
    if (optimal_max_leaf_nodes == None and lowest_mae == None) or (current_mae < lowest_mae):
        optimal_max_leaf_nodes = max_leaf_nodes
        lowest_mae = current_mae

print(f"Optimal max leaf nodes: {optimal_max_leaf_nodes}. Lowest Mean Absolute Error: {lowest_mae:,.0f}")  
    
# After tuning the model, e.g. determining the optimal max_leaf_nodes, it's time to
# finally fit it with all of the training data in order to deploy it for predicting real-world data.       
final_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=0)
final_model.fit(X, y)    
    
# Advancing on a bit from Decision Trees, another model type is Random Forest.
# Using default parameters, a RandomForestRegressor fits multiple decision trees on
# various sub-samples of the dataset, and makes predictions from the average of each 
# tree component's prediction. This improves accuracy and controls overfitting.   
# See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y) 
predicted_y = forest_model.predict(validation_X)
print(mean_absolute_error(validation_y, predicted_y)) 

# There are parameters for tuning RandomForestRegressors, 
# but they generally perform well without tuning, i.e. using defaults.
    
    
    
    
    
    