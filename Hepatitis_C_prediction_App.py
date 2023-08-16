import numpy as np
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pandas import DataFrame

from google.oauth2 import service_account
#from gsheetsdb import connect
from gspread_pandas import Spread,Client



#scope = ["https://www.googleapis.com/auth/spreadsheets"]

scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']

# Create a connection object.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=scope,
)
#conn = connect(credentials=credentials)
client = Client(scope = scope, creds = credentials)
spreadsheet_name = 'Hepathitis_C_Database'
spread = Spread(spreadsheet_name, client = client)

sh = client.open(spreadsheet_name)
worksheet_list = sh.worksheets()

st.markdown("""
<style>
.css-q16mip.e3g6aar0
{
  visibility:hidden;
}
</style>
""", unsafe_allow_html=True)

#st.write(spread.url)
#@st.cache_data(ttl=600)

def load_the_spreadsheet(spreadsheet_name):
    worksheet = sh.worksheet(spreadsheet_name)
    df = DataFrame(worksheet.get_all_records())
    return df

def update_the_spreadsheet(spreadsheet_name, dataframe):
    col = ['Age','Sex',	'ALB',	'ALP',	'ALT',	'AST',	'BIL',	'CHE',	'CHOL', 'CREA',	'GGT', 'PROT']
    #spread.df_to_sheet(dataframe[col], sheet = spreadsheet_name, index = False)
    spread.df_to_sheet(dataframe, sheet = spreadsheet_name, index = False)
    st.sidebar.info('Updated to Googlesheet')




# Load our mnodel
model_load = pickle.load(open('my_model_XGBoost_hepatitisC_new.sav', 'rb'))
X_train = pd.read_csv('X_train.csv')

scaler = StandardScaler()
# Identify numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Define a prediction function

def prediction_model(input_data):
    
    input_data_as_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_array.reshape(1,-1)

    prediction = model_load.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "0=Blood Donor", "The patient is a 0=Blood Donor"
    elif prediction[0] == 1:
        return "0s=Suspect Blood Donor" ,"The patient is a 0s=Suspect Blood Donor"
    elif prediction[0] == 2:
        return "1=Hepatitis", "The patient has 1=Hepatitis"
    elif prediction[0] == 3:
        return "2=Fibrosis" ,"The patient has 2=Fibrosis"
    elif prediction[0] == 4:
        return "3=Cirrhosis" ,"The patient has 3=Cirrhosis"


def main():

    # Give a title to the App
    st.title('Hepatitis C prediction App')


    # Getting the Input from the user
    Age = st.text_input('Age of the patient(in years)')
    Sex = st.text_input('Sex of the patient( either f(for female)/m(for male))')
    ALB = st.text_input('ALB(Albumin) value')
    ALP = st.text_input('ALP(Alkaline Phosphatase) value')
    ALT = st.text_input('ALT(ALamine aminotransferase) value')
    AST = st.text_input('AST(ASparate aminotransferase) value')
    BIL = st.text_input('BIL(Bilirubin) value')
    CHE = st.text_input('CHE Value')
    CHOL = st.text_input('CHOL(CHOlesterol) Value')
    CREA = st.text_input('CREA(CREAtinine) Value')
    GGT = st.text_input('GGT(Gamma-Glutamyl Transferase) Value')
    PROT = st.text_input('PROT Value')
    
    

    data = {
    'Age': [Age],
    'Sex': [Sex],
    'ALB': [ALB],
    'ALP': [ALP],
    'ALT': [ALT],
    'AST': [AST],
    'BIL': [BIL],
    'CHE': [CHE],
    'CHOL': [CHOL],
    'CREA': [CREA],
    'GGT': [GGT],
    'PROT': [PROT]
     }

    df_ = pd.DataFrame(data)
    df_.loc[df_["Sex"] == "m", "Sex"] = "1"
    df_.loc[df_["Sex"] == "f", "Sex"] = "0"

    col_names = df_.columns
    
    df_ = df_.apply(pd.to_numeric, errors='coerce')

    df_[numeric_cols] = scaler.transform(df_[numeric_cols])
    df_s = pd.DataFrame(df_, columns=col_names)

    opt_df = pd.DataFrame(data)
    opt_df['target'] = " "
    
 
    

    # variable of prediction

    result = ''

    if st.button('Hepatitis C test result'):
        label, result = prediction_model(df_)
        opt_df.iloc[:,-1] = label
        df = load_the_spreadsheet('Sheet1')
        new_df = pd.concat([df, opt_df], ignore_index=True)
        update_the_spreadsheet('Sheet1', new_df)

    # Display the result
    st.success(result)

if __name__ == '__main__':
    main()
