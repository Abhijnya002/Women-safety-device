
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


print("Libraries updated")


#Global Variables
seed = 7


# Set some standard parameters upfront
pd.options.display.float_format = '{:.2f}'.format
plt.style.use('ggplot')
print('pandas version ', pd.__version__)



# Load data set containing all the data from csv
df = pd.read_csv('WomenSafetyDataset.csv')


# Describe the data, Shape and how many rows and columns
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))
print(list(df.columns))
print(df['OutputClass'].value_counts(), '\n')
print(df.head(5), '\n' )
print(df.describe(), '\n' )
print(df.isnull().sum(), '\n')


df.dropna(inplace=True)
print(df.isnull().sum(), '\n')
print('Number of columns in the dataframe: %i' % (df.shape[1]))
print('Number of rows in the dataframe: %i\n' % (df.shape[0]))


pd.crosstab(df.PanicSwitch, df.OutputClass).plot(kind='bar')
plt.title('Output Label Data write with PanicSwitch')
plt.xlabel('PanicSwitch')
plt.ylabel('Total Counts')
plt.savefig('OutputClasses condition Data wrt to PanicSwitch')
plt.show()



Temp_0_25 = df["Temperature"][(df["Temperature"] >= 0) & (df["Temperature"] <= 25)]
Temp_26_35 = df["Temperature"][(df["Temperature"] >= 26) & (df["Temperature"] <= 35)]
Temp_36 = df["Temperature"][(df["Temperature"] >= 36)]

aix = ["0-25", "26-35", ">35"]
aiy = [len(Temp_0_25.values), len(Temp_26_35.values), len(Temp_36.values)]

plt.figure(figsize=(8,4))
sns.barplot(x=aix, y=aiy, palette="Set2")
plt.title("Temperature")
plt.xlabel("Value")
plt.ylabel("Range")
plt.savefig('Temperature Graph')
plt.show()


print("Pre-Processing Done")


X = df.iloc[:,1:6]
print("X = ",'\n', X)
y = df.iloc[:,-1]
print('\n', y)


train_X,test_X,train_y,test_y = train_test_split(
    X,y,test_size=0.2, random_state=seed )


print('\n train_X = \n', train_X)
print('\n test_X = \n', test_X)
print('\n train_y = \n', train_y)
print('\n test_y = \n', test_y)
final_test = test_X.iloc[0:5]
print('\n final_test = \n', final_test)


print("\n Machine Learning Model Build \n")
models=[]


#Logistic Regression
models.append(("logreg",LogisticRegression()))
LR_Model = LogisticRegression()
LR_Model.fit(train_X,train_y)
LR_Prediction = LR_Model.predict(test_X)
outp1 = LR_Model.predict(final_test)
print('\n', test_y)
print("LR Alg Accuracy", int(accuracy_score(test_y,LR_Prediction) *100), 'Percent' )
print( "Output Prediction LR = ", outp1, '\n' )



#KNN
KNN_Model = KNeighborsClassifier()
KNN_Model.fit(train_X,train_y)
KNN_Prediction = KNN_Model.predict(test_X)
outp3 = KNN_Model.predict(final_test)
print('\n', test_y[:10])
print("KNN Alg Accuracy", int(accuracy_score(test_y,KNN_Prediction) *100), "Percent" )
print( "Output Prediction KNN = ", outp3, '\n' )



print("Project End")







