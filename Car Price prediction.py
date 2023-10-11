# import requied modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score

# loading car price dataset
price_dataset = pd.read_csv("car data.csv")
# show the dataset
price_dataset.head(15)
# show dataset shape
price_dataset.shape
# show some statistical info about the dataset
price_dataset.describe()


# check if there is any none values in the dataset to make data cleaning or not
price_dataset.isnull().sum()


# labeling the dataset by converting each textual column into numeric column
le = LabelEncoder()
datatypes = price_dataset.dtypes
for Indexdatatype in range(len(datatypes)):
    if datatypes[Indexdatatype]=="object":
        price_dataset.iloc[:,Indexdatatype] = le.fit_transform(price_dataset.iloc[:,Indexdatatype])
price_dataset.head(15)


# find correlation between each feature in the dataset and other
correlation_values = price_dataset.corr()
# plot the dataset correlation
plt.figure(figsize=(7,7))
sns.heatmap(correlation_values,square=True,cbar=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap = 'Blues')


# print dataset
price_dataset.head()
plt.figure(figsize=(10,10))
# count number repetion for each group in transmission and plot it
price_dataset['Transmission'].value_counts()
sns.catplot(x='Transmission',data=price_dataset,kind='count')

# count number repetion for each group in transmission and plot it
price_dataset['Fuel_Type'].value_counts()
sns.catplot(x='Fuel_Type',data=price_dataset,kind='count')
plt.show()



# split data into input and label data
X = price_dataset.drop(columns=['Selling_Price','Car_Name'],axis=1)
Y = price_dataset['Selling_Price']
print(X)
print(Y)
# split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.7,random_state=2)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)




# create Linear Regression Model and train it
LRModel = LinearRegression()
LRModel.fit(x_train,y_train)
# make Model predict train and test data
predicted_train_LR= LRModel.predict(x_train)
predicted_test_LR= LRModel.predict(x_test)
# find the error value for test and test prediction by r squared score
r_train = r2_score(predicted_train_LR,y_train)
r_test = r2_score(predicted_test_LR,y_test)
print(r_train,r_test)

# draw relation between predicted train value and y_train
plt.title("Actual prices vs predicted prices")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_train,predicted_train_LR,marker="X",color="blue")
# draw relation between predicted test value and y_test
plt.title("Actual prices vs predicted prices")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_test,predicted_test_LR,marker="X",color="blue")




# create Lasso Model and train it
LasModel = Lasso()
LasModel.fit(x_train,y_train)
# make Model predict train and test data
predicted_train_Las= LasModel.predict(x_train)
predicted_test_las= LasModel.predict(x_test)
# find the error value for train and test prediction by r squared score
r_train = r2_score(predicted_train_Las,y_train)
r_test = r2_score(predicted_test_las,y_test)
print(r_train,r_test)

# draw relation between predicted train value and y_train
plt.title("Actual prices vs predicted prices")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_train,predicted_train_Las,marker="^",color="red")
# draw relation between predicted test value and y_test
plt.title("Actual prices vs predicted prices")
plt.xlabel("Actual prices")
plt.ylabel("predicted prices")
plt.scatter(y_test,predicted_test_las,marker="^",color="red")

