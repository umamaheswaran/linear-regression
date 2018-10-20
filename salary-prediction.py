#Linear regression with least square method
from flask import Flask
from flask import request

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib

from sqlalchemy import create_engine, select, Table, MetaData

import pandas as pd 
import numpy as np 
import json
import pickle

app = Flask(__name__)

engine = create_engine('mysql+mysqlconnector://root:root@localhost/linear')
conn = engine.connect()

#reflection
meta = MetaData()

#REST apis - Representational state transfer application programming interface.
# JSON - javascript object notation 

#SOAP based webservices - Simple object acess protocol - banking & insurance domains- security issues.
#XML

#ORM object relation mapper

@app.route('/')
def hello():
    return "Hello world!"


@app.route('/getName', methods = ['GET', 'POST'])
def getName():
    return "Hello " + request.args.get('userName')

#salary based on experience - Linear Regression   

@app.route('/tp',methods=['GET'])
def trainnpredict():
    print("request received.....")

    print("loading data file.....")
    df = pd.read_csv("SalaryData.csv") #2GB

    print("data preparation")
    train_set, test_set = train_test_split(df, test_size=0.2)
    df_copy = train_set.copy()
    test_set_full = test_set.copy()

    print(test_set)
    test_set = test_set.drop(["Salary"], axis=1)

    print("drop salary column")
    print(test_set)

    train_labels = df_copy["Salary"]
    train_set_full = train_set.copy()
    train_set = train_set.drop(["Salary"], axis=1)

    print("starting training ....")

    lr = LinearRegression()
    lr.fit(train_set, train_labels)

    print("training done....")

    usr_input = request.args.get('years')

    salary = lr.predict(np.array(float(usr_input)))

    output = salary.tolist()
    for sal in output:
        mp = {}
        mp['salary'] = sal

    with open("salary_model.pkl", "wb") as file_handler:
        pickle.dump(lr, file_handler)    

    return json.dumps(mp)


@app.route('/train',methods=['GET'])
def train():
    df = pd.read_csv("SalaryData.csv")
    train_set, test_set = train_test_split(df, test_size=0.2)
    df_copy = train_set.copy()
    test_set_full = test_set.copy()
    test_set = test_set.drop(["Salary"], axis=1)

    train_labels = df_copy["Salary"]
    train_set_full = train_set.copy()
    train_set = train_set.drop(["Salary"], axis=1)

    lr = LinearRegression()
    lr.fit(train_set, train_labels)

    with open("salary_model.pkl", "wb") as file_handler:
        pickle.dump(lr, file_handler)    

    return json.dumps('Training done')

@app.route('/predict',methods=['GET'])
def predict():
    usr_input = request.args.get('years')

    lr = joblib.load("salary_model.pkl")
    salary = lr.predict(np.array(float(usr_input)))

    output = salary.tolist()
    for sal in output:
        mp = {}
        mp['salary'] = sal

    return json.dumps(mp)  

@app.route('/predictPost',methods=['POST'])
def predictPost():
    result = []
    mp = {}
    try:  #exception handling
        usr_input = request.get_json()
        user_exp = float(usr_input["experience"])

        lr = joblib.load("salary_model.pkl")
        salary = lr.predict(np.array(float(user_exp)))

        emp = Table('emp', meta, autoload=True, autoload_with=engine)
        insData = emp.insert()
        
        output = salary.tolist()
        for sal in output:
            conn.execute(insData,experience=user_exp, salary=sal)
            mp['salary'] = sal
            result.append(mp)
    except:
        result.append("Oops, something went wrong")

    return json.dumps(result)     


if __name__ == '__main__':
    app.run(debug=True)