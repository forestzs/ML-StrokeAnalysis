import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.metrics import mean_squared_error

# Get data
df = pd.read_csv('stroke_data.csv')

# Seperate into data_train and data_test
train = df.sample(frac=0.8, random_state=1)
test = df.drop(train.index)
features = ['age','avg_glucose_level','bmi']
x_train = train[features]
y_train = train.stroke
x_test = test[features]
y_test = test.stroke

# Train the logic model
lg_model = LogisticRegression()
lg_model.fit(x_train, y_train)
y_pred = lg_model.predict(x_test)

# Predict under the condition
def predict_stroke_prob(age, avg_glucose_level, bmi):

    input_vec = [[age, avg_glucose_level, bmi]]

    return lg_model.predict_proba(input_vec)[0][1]

prob = predict_stroke_prob(age=60, avg_glucose_level=120, bmi=20)

if prob < 0:
    prob = 0;

# Prediction
print('The possibility of getting stroke：{:.2f}%'.format(prob * 100))

# MAE
y_predict = lg_model.predict(x_test)
mae = mean_absolute_error(y_test, y_predict)
print("MAE:", mae)

#MSE
MSE = mean_squared_error(y_test, y_pred)
print("MSE: ", MSE)

#RMSE
RMSE = np.sqrt(MSE)
print("RMSE: ", RMSE)

# Accurancy
accuracy = accuracy_score(y_test, [1 if x>=0.05 else 0 for x in y_pred])
print('Accurancy：{:.2f}%'.format(accuracy * 100))


