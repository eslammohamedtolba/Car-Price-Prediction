import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os


# -------------------------------------------------- Load data
df = pd.read_csv('PrepareModel\cardekho.csv')
df.head()
# Show the shape of the dataset
print(df.shape)
# Show some statistical information about dataset
df.describe()

columns_names = df.columns
print(columns_names)
print(df.info())



# ----------------------------------------------Exploratory data analysis

# Take copy from data to make processes on
preprocessed_df = deepcopy(df)

# Show all types of columns in the data
print(preprocessed_df.dtypes)

# Check about none values in data to decide if we will make data cleaning or not
print(preprocessed_df.isnull().sum())


# Handle missing values of float columns that are mileage(km/ltr/kg), engine, seats
column_float_imputed = ['mileage(km/ltr/kg)', 'engine','seats']
preprocessed_df[column_float_imputed] = preprocessed_df[column_float_imputed].fillna(preprocessed_df[column_float_imputed].mean())

print(preprocessed_df.isnull().sum())
# Handle missing values of textual column max_power
max_power_col = 'max_power'
preprocessed_df[max_power_col] = preprocessed_df[max_power_col].fillna(preprocessed_df[max_power_col].mode()[0])
# Check about missing values again
print(preprocessed_df.isnull().sum())


# Handling invalid data
# Get textual columns by select dtypes function
textual_columns = preprocessed_df.select_dtypes(include = ['object']).columns
print(textual_columns)
print(preprocessed_df[textual_columns])

# Convert textual max power columns into numerical column
spaces_count = preprocessed_df['max_power'].apply(lambda x: x.count(' ')).sum()
print(spaces_count)

# Replace each space with string zero
preprocessed_df['max_power'] = preprocessed_df['max_power'].str.replace(' ', '0')
# Convert max power column from textual into numerical
preprocessed_df['max_power'] = preprocessed_df['max_power'].astype(float)

textual_colmns = preprocessed_df.select_dtypes(include=['object'])
textual_columns_names = textual_colmns.columns
textual_colmns


# Label encoding textual columns

# Dictionary to store the encoders
label_encoders = {}

# Fit and transform each textual column, saving the encoders
for col in textual_columns_names:
    label_encoder = LabelEncoder()
    preprocessed_df[col] = label_encoder.fit_transform(preprocessed_df[col])
    label_encoders[col] = label_encoder

preprocessed_df.head()

# Check about if there is any extra textual columns
preprocessed_df.select_dtypes(include=['object']).columns


correlation = preprocessed_df.corr()
sns.heatmap(correlation, cbar=True,square=True,fmt='.2f',annot=True,annot_kws={'size':8},cmap = 'Blues')
plt.show()
# plot correlation degree between selling price column and other columns
correlation['selling_price'].drop('selling_price').sort_values(ascending = False).plot(kind = 'bar')


# Handling outliers

# Check Skewness for each column
column_skewed = preprocessed_df.columns.drop('selling_price')
skewness = preprocessed_df.drop(columns = ['selling_price']).skew()
print(skewness)
plt.figure(figsize = (15,7))
plt.bar(column_skewed, skewness)
plt.show()

# Visualize box plot for dataframe
plt.figure(figsize = (15,10))
sns.boxplot(data = preprocessed_df.drop(columns = ['selling_price']))
plt.show()

# Find skewness for km driven column alone to handle it 
preprocessed_df['km_driven'].skew()
# Handle skewness of km driven columns by taking the log function for it
preprocessed_df['km_driven'] = np.log(preprocessed_df['km_driven'])
# Find skewness for km driven column after we handled it
preprocessed_df['km_driven'].skew()
# Check Skewness for each column after the handling process
column_skewed = preprocessed_df.columns.drop('selling_price')
skewness = preprocessed_df.drop(columns = ['selling_price']).skew()
plt.figure(figsize = (15,7))
plt.bar(column_skewed, skewness)

# Show first samples of dataframe after preprocessing 
preprocessed_df.head()



categorical_columns = ['year', 'fuel','seller_type', 'transmission','owner','seats']
plt.figure(figsize = (15,12))
for index, col in enumerate(categorical_columns):
    values_count = preprocessed_df[col].value_counts()
    plt.subplot(2, 3, index + 1)
    plt.axis('off')
    plt.title(col)
    plt.pie(values_count, labels=values_count.index)
plt.show()

continuous_columns = ['km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power']
plt.figure(figsize = (15,12))
for index, col in enumerate(continuous_columns):
    plt.subplot(2, 2, index + 1)
    plt.axis('off')
    plt.title(col)
    plt.hist(preprocessed_df[col])
plt.show()

preprocessed_df.head()


# -------------------------------------------------------------Modeling

# Split data into input and label data
X = preprocessed_df.drop(columns = ['selling_price'])
Y = preprocessed_df['selling_price']
print(f'size of input data {X.shape}')
print(f'size of input data {Y.shape}')
# Split data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.12, random_state = 42)
print(f'x train size {x_train.shape}, x test size {x_test.shape}')
print(f'y train size {y_train.shape}, y test size {y_test.shape}')

# Function to visualize actual and predicted lables
def plot(y_train, y_train_pred, y_test, y_test_pred):
    plt.figure(figsize = (7,7))
    plt.title('Y train VS predicted Y train')
    plt.xlabel('Y train')
    plt.ylabel('predicted Y train')
    plt.scatter(y_train, y_train_pred, color = 'red', marker = 'X')
    plt.plot(range(int(min(y_train)), int(max(y_train))), color = 'black')
    plt.show()
    plt.figure(figsize = (7,7))
    plt.title('Y test VS predicted Y test')
    plt.xlabel('Y test')
    plt.ylabel('predicted Y test')
    plt.scatter(y_test, y_test_pred, color = 'blue', marker = 'o')
    plt.plot(range(int(min(y_test)), int(max(y_test))), color = 'black')
    plt.show()

# function to perform all functionalities of model
def model_functionality(model, x_train, y_train, x_test, y_test):
    
    # Create regression model
    reg_model = model
    reg_model.fit(x_train, y_train)
    
    # Get scores of the model on train and test data
    score_train = reg_model.score(x_train, y_train)
    score_test = reg_model.score(x_test, y_test)
    print(f'train score {score_train}, test score {score_test}')
    
    # Make model predict on train and test 
    predicted_y_train = reg_model.predict(x_train)
    predicted_y_test = reg_model.predict(x_test)
    plot(y_train, predicted_y_train, y_test, predicted_y_test)
    
    # Find model's error on train and test data using mean squared error metric
    train_error = mean_squared_error(y_train, predicted_y_train)
    test_error = mean_squared_error(y_test, predicted_y_test)
    
    print(f'train error {train_error}, test error {test_error}\n\n')
    
    return reg_model

# Create all models
models = {
    'Linear regression model': LinearRegression(), 
    'Ridge regression model': Ridge(), 
    'Lasso regression model': Lasso(), 
    'Elastic regression model': ElasticNet(), 
    'Decision tree model': DecisionTreeRegressor(),
    'Gradient boosting model': GradientBoostingRegressor(),
    'K-nearest neighbors model': KNeighborsRegressor(),
    'Random forest model': RandomForestRegressor()
}
# Train each model and plot the results
all_trained_models = {}
for name, model in models.items():
    print(name, end=':\n\n')
    trained_model = model_functionality(model, x_train, y_train, x_test, y_test)
    all_trained_models[name] = deepcopy(trained_model)

# Show all models with their names
all_trained_models



# Based on the previous results, the Random forest model is the best one on the test data with accuracy 97.6%, 
# so we will access it to analyze the previous results

# Get random forest model again to work with 
Random_forest_model = all_trained_models['Random forest model']
print(Random_forest_model)

# Find all features with its importance
importances = Random_forest_model.feature_importances_
features = x_train.columns

# combine features with its importance in one dataframe
features_with_importances = pd.DataFrame({'Features':features, 'Importance':importances})

# Sort dataframe based on importance column
features_with_importances = features_with_importances.sort_values(['Importance'], ascending = False)
features_with_importances



# -----------------------------------------Save best model and label encoders

# Paths to save the encoders and model
label_encoders_path = 'PrepareModel/label_encoders.sav'
random_forest_model_path = 'PrepareModel/Random_forest_model.sav'

# Save the dictionary of label encoders if not already saved
if not os.path.exists(label_encoders_path):
    joblib.dump(label_encoders, label_encoders_path)
    print(f'Saved label encoders to {label_encoders_path}')
else:
    print(f'Label encoders already saved at {label_encoders_path}')

# Save the trained RandomForest model if not already saved
if not os.path.exists(random_forest_model_path):
    joblib.dump(Random_forest_model, random_forest_model_path)
    print(f'Saved RandomForest model to {random_forest_model_path}')
else:
    print(f'RandomForest model already saved at {random_forest_model_path}')
