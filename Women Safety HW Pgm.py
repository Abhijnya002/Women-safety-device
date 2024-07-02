
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import serial
import os
import xlrd
import xlwt 
from xlwt import Workbook 


print("Libraries updated")


#Global Variables
ser = 0
seed = 7


# Set some standard parameters upfront
pd.options.display.float_format = '{:.2f}'.format
plt.style.use('ggplot')
print('pandas version ', pd.__version__)

def init_serial():
    COMNUM = 4          #Enter Your COM Port Number Here.
    global ser          #Must be declared in Each Function
    ser = serial.Serial('com7', 9600)
    ser.timeout = 10
    if ser.isOpen():
        print( 'Open: ' + ser.portstr)

init_serial()


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
#plt.show()



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
#plt.show()


print("Pre-Processing Done")


#open the excel
# Workbook is created 
wb = Workbook() 
# add_sheet is used to create sheet. 
sheet1 = wb.add_sheet('WomenSafety Test') 
sheet1.write(0, 0, '35') 
sheet1.write(0, 1, '1') 
sheet1.write(0, 2, '135') 
sheet1.write(0, 3, '185') 
sheet1.write(0, 4, '30') 
sheet1.write(0, 5, '73')


sheet1.write(1, 0, '60') 
sheet1.write(1, 1, '0') 
sheet1.write(1, 2, '155') 
sheet1.write(1, 3, '165') 
sheet1.write(1, 4, '26') 
sheet1.write(1, 5, '71')


##sheet1.write(2, 0, '2') 
##sheet1.write(2, 1, '0') 
##sheet1.write(2, 2, '25') 
##sheet1.write(2, 3, '68') 
##sheet1.write(2, 4, '120') 
wb.save('WomenSafety HW Test.xls')
#os.startfile("F:\Python Codes\Stress Detection with HW/Stress HW Test.xls")
print("Excel open")



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



HW_Data = bytes(1)
while True:
    print ( "Waiting for Hardware Real Time Data" )
#    data = ser.readline()  #Read from Serial Port
    data = ser.read( )
#    print(data)
    if(data == b'@'):
        print(HW_Data)
        print("Exit")
        break
            
    else:
        HW_Data = HW_Data + data
#        print(HW_Data)



print ( "Waiting" )
### T--H-S-G1---G2---G3---
##    print ( HW_Data )           # A-T--H--S---@
##    if( HW_Data == b'' ):
##        print("Wrong Data")
##
##    else:
#        print("ok")
#P0X000Y000T00H00@
#12345678901234567


HW_Data = HW_Data.decode("utf-8")
print ( HW_Data )
print ( HW_Data[2:3] )
print ( HW_Data[4:7] )
print ( HW_Data[8:11] )
print ( HW_Data[12:14] )
print ( HW_Data[15:17] )


sheet1.write(2, 0, '45') 
sheet1.write(2, 1, HW_Data[2:3]) 
sheet1.write(2, 2, HW_Data[4:7]) 
sheet1.write(2, 3, HW_Data[8:11])
sheet1.write(2, 4, HW_Data[12:14])
sheet1.write(2, 5, HW_Data[15:17])

wb.save('WomenSafety HW Test.xls')
print("Wait for prediction")
print("")
final_test_HW = pd.read_excel('WomenSafety HW Test.xls')
print( final_test_HW.head(3) )

final_test_HW1 = final_test_HW.iloc[:,1:6]
#        final_test_HW1 = final_test_HW[0:4]
print( final_test_HW1 )
outp = LR_Model.predict( final_test_HW1 )
print("")
print( "Output Prediction Hardware Data = ", outp )
print(outp[1:2])
WomenSafety = outp[1:2]
if( WomenSafety == 0 ):
    print("Person Is In Problem")
elif( WomenSafety == 1 ):
    print("Person Is Safe")
elif( WomenSafety == 2 ):
    print("Person Is in Problem With Health Issue")




print("Project End")







