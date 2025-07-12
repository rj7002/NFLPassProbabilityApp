import streamlit as st
import pandas as pd
import plotly.express as px
from random import randint
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/ryan/Desktop/FULLSTACK/FTNData.csv')
name = st.selectbox('Select a quarterback',df['passer_player_name'].unique())
routes = st.multiselect('Select routes',df['route'].unique())
defcov = st.selectbox('Select defensive coverage',['Both'] + list(df['defense_man_zone_type'].unique()))
defform = st.multiselect('Select defensive formation',df['defense_coverage_type'].unique())
form = st.multiselect('Select an offensive formation',df['offense_formation'].unique())
down = st.multiselect('Select downs',df['down'].unique())
qtr = st.multiselect('Select quarter',df['qtr'].unique())
pressure = st.checkbox('Pressure')
outofpock = st.checkbox('Out of Pocket')

df2 = df[df['passer_player_name'] == name]
if routes:
    df2 = df2[df2['route'].isin(routes)]
if defcov != 'Both':
    df2 = df2[df2['defense_man_zone_type'] == defcov]
if defform:
    df2 = df2[df2['defense_coverage_type'].isin(defform)]
if form:
    df2 = df2[df2['offense_formation'].isin(form)]
if down:
    df2 = df2[df2['down'].isin(down)]
if qtr:
    df2 = df2[df2['qtr'].isin(qtr)]
if pressure:
    df2 = df2[df2['was_pressure'] == True]
if outofpock:
    df2 = df2[df2['is_qb_out_of_pocket'] == True]
x_values = []
y_values = []
for index, row in df2.iterrows():
        
    if row['pass_location'] == 'left':
        x_values.append(randint(-26,-9))
    elif row['pass_location'] == 'middle':
        x_values.append(randint(-8,8))
    else: 
        x_values.append(randint(9,26))
    y_values.append(row['air_yards'])
fig, ax1 = plt.subplots(1, 1)

qb = df2
plt.style.use('dark_background')
# cmap='gist_heat',
sns.kdeplot(x=x_values, y=y_values, ax=ax1, cmap='viridis',shade=True, shade_lowest=False, n_levels=10)
ax1.set_xlabel('')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_ylabel('')
ax1.set_xlim(-53.3333/2, 53.3333/2)
ax1.set_ylim(-15,60)

for j in range(-15, 60-1, 1):
    ax1.annotate('--', (-3.1, j-0.5), ha='center', fontsize=10)
    ax1.annotate('--', (3.1, j-0.5), ha='center', fontsize=10)

for i in range(-10, 60, 5):
    ax1.axhline(i, c='w', ls='-', alpha=0.7, lw=1.5)

for i in range(-10, 60, 10):
    ax1.axhline(i, c='w', ls='-', alpha=1, lw=1.5)

for i in range(10, 60-1, 10):
    ax1.annotate(str(i), (-12.88, i-1.15), ha='center', fontsize=15, rotation=270)
    ax1.annotate(str(i), (12.88, i-0.65), ha='center', fontsize=15, rotation=90)
ax1.axhline(0, c='blue', ls='-', alpha=0.7, lw=1.5)

st.pyplot(fig)  # Use st.pyplot for Matplotlib figures

from sklearn.preprocessing import LabelEncoder


le_defense_man_zone = LabelEncoder().fit(df['defense_man_zone_type'].dropna())
le_defense_coverage = LabelEncoder().fit(df['defense_coverage_type'].dropna())
le_offense_formation = LabelEncoder().fit(df['offense_formation'].dropna())
le_route = LabelEncoder().fit(df['route'].dropna())
le_pass_location = LabelEncoder().fit(df['pass_location'].dropna())

# Streamlit UI
st.sidebar.title("Pass Completion Model")
input_features = {}

# Numeric Inputs
input_features['n_offense_backfield'] = st.sidebar.number_input('Number of Offense Backfield', min_value=0, step=1)
input_features['time_to_throw'] = st.sidebar.number_input('Time to Throw (seconds)', min_value=0.0, step=0.1)
input_features['number_of_pass_rushers'] = st.sidebar.number_input('Number of Pass Rushers', min_value=0, step=1)
input_features['n_blitzers'] = st.sidebar.number_input('Number of Blitzers', min_value=0, step=1)
input_features['defenders_in_box'] = st.sidebar.number_input('Defenders in Box', min_value=0, step=1)
input_features['yardline_100'] = st.sidebar.slider('Yardline (100 - distance to endzone)', 1, 100)
input_features['down'] = st.sidebar.selectbox('Down', options=[1, 2, 3, 4])
input_features['ydstogo'] = st.sidebar.number_input('Yards to Go', min_value=1, step=1)
input_features['ngs_air_yards'] = st.sidebar.number_input('Air Yards (Next Gen Stats)', min_value=0.0, step=0.1)
input_features['qtr'] = st.sidebar.selectbox('Quarter', options=[1, 2, 3, 4])

# Encoded Categoricals
selected_man_zone = st.sidebar.selectbox('Defense Man/Zone Type', le_defense_man_zone.classes_)
input_features['defense_man_zone_type_encoded'] = le_defense_man_zone.transform([selected_man_zone])[0]

selected_coverage = st.sidebar.selectbox('Defense Coverage Type', le_defense_coverage.classes_)
input_features['defense_coverage_type_encoded'] = le_defense_coverage.transform([selected_coverage])[0]

selected_formation = st.sidebar.selectbox('Offense Formation', le_offense_formation.classes_)
input_features['offense_formation_encoded'] = le_offense_formation.transform([selected_formation])[0]

selected_route = st.sidebar.selectbox('Route', le_route.classes_)
input_features['route_encoded'] = le_route.transform([selected_route])[0]

selected_pass_loc = st.sidebar.selectbox('Pass Location', le_pass_location.classes_)
input_features['pass_location_encoded'] = le_pass_location.transform([selected_pass_loc])[0]

# Binary switches
binary_features = [
    'is_play_action', 'is_rpo', 'is_qb_out_of_pocket',
    'is_trick_play', 'is_motion',
    'was_pressure', 'no_huddle','is_contested_ball','qb_hit','qb_scramble'
]

for feature in binary_features:
    input_features[feature] = int(st.sidebar.checkbox(feature.replace('_', ' ').capitalize()))

# Prediction
input_df = pd.DataFrame([input_features])
features_order = [
'n_offense_backfield', 'time_to_throw', 'number_of_pass_rushers', 
'n_blitzers', 'defenders_in_box', 'yardline_100', 
'down', 'ydstogo', 'defense_man_zone_type_encoded', 'defense_coverage_type_encoded', 
'is_play_action', 'is_rpo', 'is_qb_out_of_pocket', 'is_trick_play',
'is_motion', 'was_pressure','ngs_air_yards','route_encoded','no_huddle','qtr','qb_hit',
'offense_formation_encoded','is_contested_ball','pass_location_encoded','qb_scramble'
]

# Reorder columns to match training
input_df = input_df[features_order]
import joblib
rfc = joblib.load('/Users/ryan/Desktop/FULLSTACK/pass_completion_model.pkl')
scaler = joblib.load('/Users/ryan/Desktop/FULLSTACK/scaler.pkl')
input_df_scaled = scaler.transform(input_df)  # use the same scaler as before
prediction = rfc.predict_proba(input_df_scaled)

st.sidebar.write("Completion Probability:", str((round(prediction[0][1],4)*100)), "%")
