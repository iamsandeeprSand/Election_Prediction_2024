import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset

def load_data():
    df = pd.read_csv('election_results.csv')
    df['Year'] = df['Year'].astype(int)
    label_encoder_st_name = LabelEncoder()
    label_encoder_pc_name = LabelEncoder()
    label_encoder_partyabbre = LabelEncoder()
    label_encoder_cand_sex = LabelEncoder()

    df['State_Name'] = label_encoder_st_name.fit_transform(df['State_Name'])
    df['PC_Name'] = label_encoder_pc_name.fit_transform(df['PC_Name'])
    #df['pc_type'] = label_encoder_pc_type.fit_transform(df['pc_type'])
    df['Sex'] = label_encoder_cand_sex.fit_transform(df['Sex'])
    df['Party'] = label_encoder_partyabbre.fit_transform(df['Party'])

    # Inverse transform for display
    df['st_name_display'] = label_encoder_st_name.inverse_transform(df['State_Name'])
    df['pc_name_display'] = label_encoder_pc_name.inverse_transform(df['PC_Name'])
    #df['pc_type_display'] = label_encoder_pc_type.inverse_transform(df['pc_type'])
    df['cand_sex_display'] = label_encoder_cand_sex.inverse_transform(df['Sex'])
    df['partyabbre_display'] = label_encoder_partyabbre.inverse_transform(df['Party'])

    return df, label_encoder_st_name, label_encoder_pc_name, label_encoder_cand_sex, label_encoder_partyabbre

df, label_encoder_st_name, label_encoder_pc_name, label_encoder_cand_sex, label_encoder_partyabbre = load_data()

# Load the model
def load_prediction_model():
    return load_model('election_prediction_model.h5')

model = load_prediction_model()

# EDA
st.title('Election Result Prediction & Data Analysis')

# Gender representation over the years
st.subheader('Gender Representation Over the Years')
gender_trend = df.groupby('Year')['cand_sex_display'].value_counts().unstack().fillna(0)
gender_trend.plot(kind='bar', stacked=True, figsize=(15, 7))
st.pyplot(plt)

# Prediction for 2024
st.subheader('Predict Election Results for 2024')

# Create a form for user input
with st.form('prediction_form'):
    year = 2024
    st_name = st.selectbox('State Name', df['st_name_display'].unique())

    # Filter AC names and numbers based on the selected state
    filtered_df = df[df['st_name_display'] == st_name]
    #pc_no = st.selectbox('Assembly Constituency Number', filtered_df['PC_No'].unique())
    #con_no = st.selectbox('Assembly Constituency Number', filtered_df['Constituency_No'].unique())
    pc_name = st.selectbox('Assembly Constituency Name', filtered_df['pc_name_display'].unique())
    #pc_type = st.selectbox('Assembly Constituency Type', filtered_df['pc_type_display'].unique())
    cand_sex = st.selectbox('Candidate Gender', df['cand_sex_display'].unique())
    partyabbre = st.selectbox('Party Abbreviation', filtered_df['partyabbre_display'].unique())
        
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Encode the selected values for prediction
        st_name_encoded = label_encoder_st_name.transform([st_name])[0]
        pc_name_encoded = label_encoder_pc_name.transform([pc_name])[0]
        #pc_type_encoded = label_encoder_pc_type.transform([pc_type])[0]
        cand_sex_encoded = label_encoder_cand_sex.transform([cand_sex])[0]
        partyabbre_encoded = label_encoder_partyabbre.transform([partyabbre])[0]
        
        input_data = np.array([[ st_name_encoded, year, pc_name_encoded, partyabbre_encoded, cand_sex_encoded]])
        prediction = model.predict(input_data)
        probability_of_winning = prediction[0][0] * 100
        st.write(f'Prediction: Approximate Vote percentage of {partyabbre} in this constituency is: {probability_of_winning:.2f}%')
        st.write(f'Raw Prediction Value: {prediction[0][0]}')

        # Load the data
df = pd.read_csv('election_results.csv')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Normalize the 'Party' column to ensure consistency
df['Party'] = df['Party'].str.upper()

# Filter the data for BJP and INC
df_bjp = df[df['Party'] == 'BJP']
df_inc = df[df['Party'] == 'INC']

# Combine the filtered data
df_combined = pd.concat([df_bjp, df_inc])

# Plot using Seaborn for better aesthetics
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_combined, x='Year', y='Total_Votes', hue='Party', style='Party', markers=True, dashes=False)
plt.xlabel('Year')
plt.ylabel('Total Votes')
plt.title('Total Votes by Year for BJP and INC')
plt.legend(title='Party')
plt.grid(True)

# Display plot using Streamlit
st.pyplot()

# Plot using Plotly for interactivity
fig = px.line(df_combined, x='Year', y='Total_Votes', color='Party', title='Total Votes by Year for BJP and INC',
              labels={'Total_Votes': 'Total Votes', 'Year': 'Year'})
st.plotly_chart(fig)
