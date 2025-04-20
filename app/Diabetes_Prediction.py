import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from random import seed
from random import gauss
from matplotlib import pyplot as plt
from sklearn import metrics
#from collections import Counter



class LR:
    
    def __init__(self,lr=0.001,n_iters=100):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    
    def _sigmoid(self,z):
        return 1.0 / (1 + np.exp(-z))
    
    def predict(self,features):
        
        linear_model = np.dot(features, self.weights)+self.bias
        y_predicted=self._sigmoid(linear_model)
        y_predicted_cls=[1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls,y_predicted
       
    def cost_function(self,features, labels,n_samples):
        
        predictions = self._sigmoid(np.dot(features,self.weights)+self.bias)
        ol=list(self.weights)
        theta=0
        for i in ol:
            theta+=i*i
        
        cost=((-labels*np.log(predictions))-((1-labels)*np.log(1-predictions)))
        cost = cost.sum() / n_samples
    
        return cost
    
    
    def update_weights(self,features,labels,n_samples):
        
        linear_model=np.dot(features,self.weights)+self.bias  
        y_predicted=self._sigmoid(linear_model)
        '''ol=list(self.weights)
        theta=0
        for i in ol:
            theta+=i'''
            
        gradient=(1 / n_samples)*(np.dot(features.T,(y_predicted-labels)))
        db=(1 / n_samples)*(np.sum(y_predicted-labels))
        self.weights-=self.lr*gradient
        self.bias-=self.lr*db
        
        
    def train(self,features, labels):
        n_samples,n_features=features.shape
        v=[]
        seed(2)
        for _ in range(n_features):
            v.append(gauss(0,1))
        self.weights=np.array(v)
        self.bias=0
        #print('self.wei',self.weights)
        cost_history = []
    
        for i in range(self.n_iters):
            self.update_weights(features, labels,n_samples)
            cost = self.cost_function(features, labels,n_samples)
            cost_history.append(cost)
            if i % 1000 == 0:
                print("iter: "+str(i) + " cost: "+str(cost))
                
        return cost_history
            
        
    
    
    
    def accuracy(self,X_test,y_true):
        y_pred,y_prob=self.predict(X_test)
        plt.figure(figsize=(10,3))
        yp=list(y_prob)
        yt=list(y_true)
        
        ar_true_0=[i for i in range(len(yt)) if yt[i]==0]
        ar_true_1=[i for i in range(len(yt)) if yt[i]==1]
        ar_pred_0=[yp[j] for j in ar_true_0]
        ar_pred_1=[yp[j] for j in ar_true_1]
       
        plt.hist(ar_pred_0,bins=50,label='Negatives',color='r')
        plt.hist(ar_pred_1,bins=50,label='Positiives',alpha=0.5,color='b')
        plt.xlabel('prob of being positive')
        plt.ylabel('num of records')
        plt.legend(fontsize=10)
        plt.tick_params(axis='both',labelsize=10)
        plt.show()
        
        conf_matrix=metrics.confusion_matrix(y_true,y_pred)
        print('confusion matrix')
        print(conf_matrix)
        accuracy=np.sum(y_true==y_pred)/len(y_true)
        return accuracy,conf_matrix
        
#reading csv file  
df=pd.read_csv('C:\\Users\\Ayyappa\\OneDrive\\Documents\\ganesh\\diabetes_project.csv')


df.reset_index(drop=True,inplace=True)
print(df)

#data cleaning
filt=(df['Outcome']==0) & (df['Insulin']!=0)
ins_med_0=df.loc[filt].median()
#print('is',ins_med_0['Insulin'])
df.loc[((df.Insulin==0) & (df.Outcome==0)) ,'Insulin']=130.2


filt=(df['Outcome']==1) & (df['Insulin']!=0)
ins_med_1=df.loc[filt].median()
#print('is',ins_med_1['Insulin'])
df.loc[((df.Insulin==0) & (df.Outcome==1)) ,'Insulin']=206.8


filt=(df['Outcome']==0) & (df['Glucose']!=0)
glu_med_0=df.loc[filt].mean()
#print('gl0',glu_med_0['Glucose'])
df.loc[((df.Glucose==0) & (df.Outcome==0)) ,'Glucose']=110.0

filt=(df['Outcome']==1) & (df['Glucose']!=0)
glu_med_1=df.loc[filt].mean()
#print('gl1',glu_med_1['Glucose'])
df.loc[((df.Glucose==0) & (df.Outcome==1)) ,'Glucose']=142.3


filt=(df['Outcome']==0) & (df['SkinThickness']!=0)
sk_med_0=df.loc[filt].mean()
#print('sk_med_0',sk_med_0['SkinThickness'])
df.loc[((df.SkinThickness==0) & (df.Outcome==0)) ,'SkinThickness']=27.0

filt=(df['Outcome']==1) & (df['SkinThickness']!=0)
sk_med_1=df.loc[filt].mean()
#print('sk1',sk_med_1['SkinThickness'])
df.loc[((df.SkinThickness==0) & (df.Outcome==1)) ,'SkinThickness']=33.0

filt=(df['Outcome']==0) & (df['BloodPressure']!=0)
bp_med_0=df.loc[filt].mean()
#print('bp_med_0',bp_med_0['BloodPressure'])
df.loc[((df.BloodPressure==0) & (df.Outcome==0)) ,'BloodPressure']=70.0

filt=(df['Outcome']==1) & (df['BloodPressure']!=0)
bp_med_1=df.loc[filt].mean()
#print('bp1',bp_med_1['BloodPressure'])
df.loc[((df.BloodPressure==0) & (df.Outcome==1)) ,'BloodPressure']=74.5


'''
print('upd',df[['Insulin','Glucose','SkinThickness','Outcome']])
print('insulin',df['Insulin'].value_counts())
print('Glucose',df['Glucose'].value_counts())
print('SkinThickness',df['SkinThickness'].value_counts())
print('BloodPressure',df['BloodPressure'].value_counts())
#df.to_excel('preprocessed.xlsx')
'''



bc=df.loc[:,['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
dc=df.loc[:,['Outcome']]


X=bc.values.tolist()
X=np.array(X)
y=dc.values.tolist()
y=np.array(y).flatten()

#feature scaling
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X_norm=(X-mean)/std


#datasplitting
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=6)
reg=LR(lr=0.01,n_iters=20000)
cost=reg.train(X_train,y_train)       

#evaluation
test_accuracy,conf_matrix=reg.accuracy(X_test,y_test)


TP=conf_matrix[1][1]
TN=conf_matrix[0][0]
FP=conf_matrix[0][1]
FN=conf_matrix[1][0]

print('tp=',TP,'fn=',FN,'fp=',FP,'tn=',TN)
print('actual pos tuples',TP+FN)
print('actual neg tuples',TN+FP)

print('weights',reg.weights)
print('bias',reg.bias)
error_rate=(FP+FN)/(TP+TN+FP+FN)
acc=(TP+TN)/(TP+TN+FP+FN)
sensitivity_recall=(TP)/(TP+FN)
specificity=(TN)/(FP+TN)
precision=(TP)/(TP+FP)
F1score=(2*precision*sensitivity_recall)/(precision+sensitivity_recall)
print('--------------------------------')
print('accuracy=',np.round(acc*100,2))
print('error_rate=',np.round(error_rate*100,2))
print('sensitivity_recall=',np.round(sensitivity_recall*100,2))
print('specificity=',np.round(specificity*100,2))
print('precision=',np.round(precision*100,2))
print('F1score=',np.round(F1score*100,2))
print('---------------------------------')


#cost function graph
xaxis=list(range(reg.n_iters))
yaxis=cost


plt.plot(xaxis,yaxis,color='red')
plt.xlabel('iterations')
plt.tight_layout()
plt.style.use('seaborn')
plt.ylabel('cost value')
plt.title('COST FUNCTION')
plt.show()


