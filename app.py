import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# load trained model
model = tf.keras.models.load_model('model.h5')

# load encoder and scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_country.pkl','rb') as file:
    onehot_encoder_country=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# streamlit application


# title
st.title('Customer Churning Prediction')

#markdown
st.markdown("""Welcome to Customer Churn Prediction Dashboard.
This application uses AI to predict whether a customer will churn or not.
""")

# user input
country = st.selectbox('country',onehot_encoder_country.categories_[0])
gender = st.selectbox('gender', label_encoder_gender.classes_)
credit_card = st.selectbox('credit_card',[0,1])
active_member = st.selectbox('active_member',[0,1])
balance = st.number_input('balance')
credit_score = st.number_input('credit_score')
estimated_salary = st.number_input('estimated_salary')
age = st.slider('age',18,92)
tenure = st.slider('tenure',0,10)
products_number = st.slider('products_number',1,4)


# Prepare input data
input_data = pd.DataFrame({
    'credit_score': [credit_score],
    'gender': [label_encoder_gender.transform([gender])[0]],
    'age': [age],
    'tenure':[tenure],
    'balance':[balance],
    'products_number':[products_number],
    'credit_card':[credit_card],
    'active_member':[active_member],
    'estimated_salary':[estimated_salary]
})

# onehot encoding country
country_encoded = onehot_encoder_country.transform([[country]])
country_encoded_df = pd.DataFrame(country_encoded, columns = onehot_encoder_country.get_feature_names_out(['country']))
input_data = pd.concat([input_data.reset_index(drop=True), country_encoded_df], axis=1)

# scaling
input_data_scaled = scaler.transform(input_data)

# make predictions
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]


# Display churn probability
st.write(f'Churn Probability: {prediction_prob:.2f}')

# Plotly bar chart to visualize the probability
import plotly.graph_objects as go
fig = go.Figure(go.Bar(
    x=["Churn", "No Churn"],
    y=[prediction_prob, 1 - prediction_prob],
    marker_color=['#ff4b4b', '#4caf50'],
))

fig.update_layout(
    title="Churn Probability Visualization",
    xaxis_title="Prediction",
    yaxis_title="Probability",
)

# Display the plot
st.plotly_chart(fig)


# Enhance the result with color-coded messages
if prediction_prob > 0.5:
    st.markdown(
        '<div style="background-color: #ffcccc; padding: 10px; border-radius: 10px; color: red; font-size: 20px;">'
        'Customer is likely to churn</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="background-color: #d4edda; padding: 10px; border-radius: 10px; color: green; font-size: 20px;">'
        'Customer is not likely to churn</div>',
        unsafe_allow_html=True
    )

    
   
























