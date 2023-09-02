import streamlit as st
import math
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

def output(week,temp,mini_temp=28.41,maxi_temp=28.47,p=1006.08,sea_lvl=1006.05,grnd_level=981.35,humidity1=33.13,clouds1=26.78,wind1_speed=2.77):
    inp=np.array([[temp],[mini_temp],[maxi_temp],[p],[sea_lvl],[grnd_level],[humidity1],[clouds1],[wind1_speed],[week]])
    inp=inp.reshape(1,-1)
    data = pd.read_csv("Delhi_Weather_data.csv")
    del data["dt_txt"]
    del data["day"]
    del data["time_of_record"]
    desc_n = pd.read_csv("desc_n.csv")
    data1 = pd.concat([data, desc_n], axis=1)
    weeks = pd.read_csv("weeks.csv")
    data1 = pd.concat([data1,weeks], axis=1)
    del data1['wind_degree']
    del data1['main']
    del data1['description']
    del data1['date']
    desc = data1['desc_n']
    features = data1.drop('desc_n', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features , desc, test_size=0.3, random_state=10)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    pre = rf.predict(X_test)
    e=rf.predict(inp)
    e=math.floor(e)
    if e==1:
        st.subheader("The predicted weather is: clear sky")
    elif e==2:
        st.subheader("The predicted weather is: few clouds")
        w="few clouds"
    elif e==3:
        st.subheader("The predicted weather is: broken clouds")
        w="broken clouds"
    elif e==4:
        st.subheader("The predicted weather is: scattered clouds")
        w="Scattered clouds"
    elif e==5:
        st.subheader("The predicted weather is: overcast clouds")
        w="overcast clouds"
    elif e==6:
        st.subheader("The predicted weather is: light rain")
        w="Light Rain"
    elif e==7:
        st.subheader("The predicted weather is: moderate rain")
        w="Moderate Rain"
    elif e==8:
        st.subheader("The predicted weather is: heavy intensity rain")
        w="Heavy intensity Rain "

def show_predict_page():
    st.title("Weather prediction")
    temp=st.number_input("Enter Temperature in Celsius (mandatory)",key="1")
    mini_temp=st.number_input("Enter Minimum Temperature in Celsius(Enter -1 for using default value)",key="2")
    maxi_temp=st.number_input("Enter Maximum Temperature in Celsius(Enter -1 for using default value)",key="3")
    p=st.number_input("Enter Pressure in torr (Enter -1 for using default value)",key="4")
    sea_lvl=st.number_input("Enter Sea Level (Enter -1 for using default value)",key="5")
    grnd_level=st.number_input("Enter Ground Level (Enter -1 for using default value)",key="6")
    humidity1=st.number_input("Enter Humidity (Enter -1 for using default value)",key="7")
    clouds1=st.number_input("Enter cloud quantity (Enter -1 for using default value)",key="9")
    wind1_speed=st.number_input("Enter wind speed (Enter -1 for using default value)",key="10")
    week=st.number_input("Enter week number (mandatory)",key="11")

    button=st.button("Find the weather")
    if button:
        for i in [mini_temp,maxi_temp,p,sea_lvl,grnd_level,humidity1,clouds1,wind1_speed]:
            if int(i)==-1:
                i=""
        output(week,temp,mini_temp,maxi_temp,p,sea_lvl,grnd_level,humidity1,clouds1,wind1_speed)
        

show_predict_page()