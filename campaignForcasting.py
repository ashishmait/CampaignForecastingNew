#!flask/bin/python
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from flask import Flask, jsonify, request,json

app = Flask(__name__)

df = pd.read_csv('CS_Complete_24Apr.csv')

feature_X_All = ['RewardID','TravelProductID' ,'IsPublic','RewardType','RewardEarnLimitCnt','PurchaseAmtMin','PurchaseAmtMax','RewardAmt','RewardPct','hotelCampaign',
                 'days','Include','Exclude','GPG','Redemption']

feature_X = ['RewardID','TravelProductID' ,'IsPublic','RewardType','RewardEarnLimitCnt','PurchaseAmtMin','PurchaseAmtMax','RewardAmt','RewardPct','hotelCampaign',
             'days','Include','Exclude','GPG']

feature_y = ['Redemption']

X_ALL = df[feature_X_All]
X = df[feature_X]
y = df[feature_y]

scaler = StandardScaler()
#scaler.fit(X)

n_neighbors = 10

suggestions = []

test_data = []
test_data_transform = []

test_isPublic = []
test_rewardType = []
test_earnLimit = []
test_Min_Amount = []
test_Max_Amount = []
test_amount = []
test_rewardPct =[]
test_hotelCampaign = []
test_Include = []
test_Exclude = []
test_Gpg =[]
test_Days = []

y_predict = []

def initialisingTestData(data):
    #Initialising test_data
    global test_data
    test_data = np.array(data)


    global test_isPublic
    test_isPublic = test_data[0,1:2]
    global test_rewardType
    test_rewardType = test_data[0,2:3]
    global test_earnLimit
    test_earnLimit = test_data[0,3:4]
    global test_Min_Amount
    test_Min_Amount = test_data[0,4:5]
    global test_Max_Amount
    test_Max_Amount = test_data[0,5:6]
    global test_amount
    test_amount = test_data[0,6:7]
    global test_rewardPct
    test_rewardPct = test_data[0,7:8]
    global test_hotelCampaign
    test_hotelCampaign = test_data[0,8:9]
    global test_Days
    test_Days = test_data[0,9:10]
    global test_Include
    test_Include = test_data[0,10:11]
    global test_Exclude
    test_Exclude = test_data[0,11:12]
    global test_Gpg
    test_Gpg = test_data[0,12:13]
    

#Start predicting values
def get_Predictions():
    global y_predict
    global suggestions
    global test_data
    tpid = test_data[0,0:1]
    
    NEW_X_ALL = X_ALL
    #[X_ALL.TravelProductID  == tpid[0]]
    X_New = NEW_X_ALL[feature_X]
    y_New = NEW_X_ALL[feature_y]
 
    new_feature_X = ['IsPublic','RewardType','RewardEarnLimitCnt','PurchaseAmtMin','PurchaseAmtMax','RewardAmt','RewardPct','hotelCampaign',
             'days','Include','Exclude','GPG']

    X_New = X_New[new_feature_X]
    #print("XNew : " + repr(X_New))

    scaler.fit(X_New)
    X_New_Transform = scaler.transform(X_New)

    X_Input =  test_data[0,1:13]
    #print("Xinput is " + repr(X_Input))

    global test_data_transform
    test_data_transform = scaler.transform(X_Input)

    neighbourCount = n_neighbors
    if(X_New.shape[0] < n_neighbors):
        neighbourCount = X_New.shape[0]

    clf = neighbors.KNeighborsRegressor(neighbourCount, weights='distance')
    clf = clf.fit(X_New_Transform, y_New.values.ravel())

    y_predict = clf.predict(test_data_transform)
    print("predicted value is : " + repr(y_predict))

    distances, indices = clf.kneighbors(test_data_transform);

    #suggestions = []

    print("Distances : " + repr(distances))
    print("Indexes : " + repr(indices))

    for i in range(0,indices[0].size):
        index = indices[0][i]
        neibhbour_redemption  = np.array(y_New[index:index+1])
        
        if(y_predict < neibhbour_redemption):
            neibhbour = scaler.inverse_transform(X_New_Transform[index:index+1])
            print("neigbhour is :" + repr(neibhbour) + " with neighbour redemption : " + repr(neibhbour_redemption))

            neighbourRewardType = neibhbour[0,1:2]
            diffRewardType = neighbourRewardType - test_rewardType

            neighbourEarnLimit = neibhbour[0,2:3]
            diffEarnLimit = neighbourEarnLimit - test_earnLimit

            neighbourMinAmount = neibhbour[0,3:4]
            diffMinAmount = neighbourMinAmount - test_Min_Amount

            neighbourMaxAmount = neibhbour[0,4:5]
            diffMaxAmount = neighbourMaxAmount - test_Max_Amount

            neighbourAmount = neibhbour[0,5:6]
            diffRewardAmount = neighbourAmount - test_amount

            neighbourPct = neibhbour[0,6:7]
            diffRewardPct = neighbourPct - test_rewardPct

            neighbourDays = neibhbour[0,8:9]
            diffDays = neighbourDays - test_Days

            neighbourGPG = neibhbour[0,11:12]
            diffRewardGpg = neighbourGPG - test_Gpg
            print("diff GPG : " + repr(diffRewardGpg))

            

            if(diffRewardType != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[1:2] = neighbourRewardType
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Reward Type from " + repr(test_rewardType) + " to " + repr(neighbourRewardType)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffEarnLimit != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[2:3] = neighbourEarnLimit
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Earn limit from " + repr(test_earnLimit) + " to " + repr(neighbourEarnLimit)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffMinAmount != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[3:4] = neighbourMinAmount
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Purchase Min Amnt from " + repr(test_Min_Amount) + " to " + repr(neighbourMinAmount)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffMaxAmount != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[4:5] = neighbourMaxAmount
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Purchase Max Amnt from " + repr(test_Max_Amount) + " to " + repr(neighbourMaxAmount)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffRewardAmount != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[5:6] = neighbourAmount
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Reward Amount from " + repr(test_amount) + " to " + repr(neighbourAmount)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffRewardPct != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[6:7] = neighbourPct
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Reward Pct from " + repr(test_rewardPct) + " to " + repr(neighbourPct)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffRewardGpg != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[11:12] = neighbourGPG
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change GPG Value from " + repr(test_Gpg) + " to " + repr(neighbourGPG)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")

            if(diffDays != 0): 
                         newTestData = np.array(X_Input)
                         newTestData[8:9] = neighbourDays
                         new_test_data_transform = scaler.transform(newTestData)
                         y_predict_new = clf.predict(new_test_data_transform);
                         if(int(y_predict_new) > int(y_predict)):
                             suggestions.append("change Campaign days from " + repr(test_Days) + " to " + repr(neighbourDays)+
                                               ", " + "redemption will Increase from " + repr(y_predict) + " to " 
                                                + repr(y_predict_new)+ "\n")      


@app.route('/campaign/api/v1.0/predictRedemption', methods=['POST'])
def get_PredictedValue():
    
    data = []

    requestData = []
    requestData.append(request.json['TPID'])
    requestData.append(request.json['IsPublic'])
    requestData.append(request.json['RewardType'])
    requestData.append(request.json['RewardEarnLimitCnt'])
    requestData.append(request.json['PurchaseAmtMin'])
    requestData.append(request.json['PurchaseAmtMax'])
    requestData.append(request.json['RewardAmt'])
    requestData.append(request.json['RewardPct'])
    requestData.append(request.json['hotelCampaign'])
    requestData.append(request.json['days'])
    requestData.append(request.json['Include'])
    requestData.append(request.json['Exclude'])
    requestData.append(request.json['GPG'])
    

    data.append(requestData)

    initialisingTestData(data)

    get_Predictions()
    return jsonify({'redemptionValue': y_predict.tolist()})


@app.route('/campaign/api/v1.0/suggestions', methods=['POST'])
def get_Suggestions():
    data = []

    requestData = []
    requestData.append(request.json['TPID'])
    requestData.append(request.json['IsPublic'])
    requestData.append(request.json['RewardType'])
    requestData.append(request.json['RewardEarnLimitCnt'])
    requestData.append(request.json['PurchaseAmtMin'])
    requestData.append(request.json['PurchaseAmtMax'])
    requestData.append(request.json['RewardAmt'])
    requestData.append(request.json['RewardPct'])
    requestData.append(request.json['hotelCampaign'])
    requestData.append(request.json['days'])
    requestData.append(request.json['Include'])
    requestData.append(request.json['Exclude'])
    requestData.append(request.json['GPG'])

    global suggestions
    
    suggestions = []

        
    print(type(suggestions))

    data.append(requestData)

    initialisingTestData(data)


    get_Predictions()
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)
