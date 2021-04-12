# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 02:20:39 2021

@author: user
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd

log_model=pickle.load(open('log_model.pkl','rb'))
dsc_model=pickle.load(open('des_model.pkl','rb'))
rand_model=pickle.load(open('Random_forest_model.pkl','rb'))

           

def pred_iris(algo_name,SepalLength,SepalWidth,PetalWidth,PetalLength):
    inputs=np.array([[SepalLength,SepalWidth,PetalWidth,PetalLength]]).astype(np.float64)
    if algo_name=='Logistic Regression':
        log_pred=log_model.predict_proba(inputs)
        return log_pred
    if algo_name=='Decision Tree Classifier':
        dsc_pred=dsc_model.predict_proba(inputs)
        return dsc_pred
    if algo_name=='Random Forest Classifier':
        rand_pred=rand_model.predict_proba(inputs) 
        return rand_pred





# Deployment time Baby

def main():
    df=pd.read_csv('G:/Study Material/Projects/Iris Flower Dataset/Classification/Iris.csv')
    
    
    st.title("Iris Classification")
    
    st.title("Lets see how our dataset looks-like")
    st.write(df)
    
    classfier_name=st.selectbox("Select Classifier",["Logistic Regression",
                                      "Decision Tree Classifier",
                                      "Random Forest Classifier"])
    st.write("You Selected",classfier_name)
    
    
    SepalLengthcm=st.slider("SepalLengthCm",4.3,7.9)
    SepalWidthcm=st.slider("SepalWidthCm",2.0,4.4)
    PetalLengthcm=st.slider("PetalLengthCm",1.0,6.9)
    PetalWidthcm=st.slider("PetalWidthCm",0.1,2.5)
    
    if st.button("Predict"):
            res=pred_iris(classfier_name,SepalLengthcm,SepalWidthcm,PetalLengthcm,PetalWidthcm) 
            list1=['Iris-setosa','Iris-versicolor','Iris-virginica'] 
            ans=list1[np.argmax(res)]
            
            st.success(ans)
    
    
if __name__=='__main__':
    main()