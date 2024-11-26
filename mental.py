import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import streamlit as st
import time
import joblib


# Load the dataset
df = pd.read_csv(r"C:\Users\sahil\Untitled Folder 1\Mental disorder symptoms (1).csv")

# Explore the dataset
print(df.head())
print(df.info())

#changing names
df = df.rename(columns={'ag+1:629e':'age'})
df = df.rename(columns={'having.trouble.in.sleeping':'trouble.sleeping'})
df = df.rename(columns={'having.trouble.with.work':'trouble.with.work'})
df = df.rename(columns={'having.nightmares':'nightmares'})
df.set_index(['age'])

df.fillna(0, inplace=True)

# Split the dataset into features and target variable
X = df.drop('Disorder', axis=1)
y = df['Disorder']

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)


# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
# Calculate the ROC AUC score for multi-class classification
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
print("ROC AUC Score:", roc_auc)


# Save the best model and the scaler
best_model = models['Random Forest']
joblib.dump(best_model, 'mental_disorder_predictor.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_target, 'label_encoder.pkl')


print(y)

st.set_page_config(page_title='Mental Disorder Predictor', layout='wide', page_icon="🧠")

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes glow {
    0% { box-shadow: 0 0 10px rgba(255, 0, 0, 0.7); }
    50% { box-shadow: 0 0 20px rgba(0, 255, 0, 0.7); }
    100% { box-shadow: 0 0 10px rgba(0, 0, 255, 0.7); }
}

h1 {
    animation: fadeIn 2s;
    color: #3D5A80;
    text-align: center;
}

input, select {
    animation: fadeIn 2s;
    margin-bottom: 20px;
}

.css-1aumxhk {
    background: #edf2f4;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    animation: glow 2s infinite alternate;
}

.stButton>button {
    background-color: #3D5A80;
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
    animation: fadeIn 3s;
}

.stButton>button:hover {
    background-color: #293241;
}
</style>
""", unsafe_allow_html=True)

# Function to display the input form
def display_input_page():
    st.title('Mental Disorder Predictor 🧠')
    st.image("https://media.giphy.com/media/l41YtZOb9EUABnuqA/giphy.gif", width=300)
    st.markdown("## Please provide your details and symptoms")
    
    user_input = {}
    
    user_input['age'] = st.selectbox(
        'Age', 
        options=[None] + list(range(int(df['age'].min()), int(df['age'].max() + 1))),
        help="Select your age from the dropdown"
    )
    
    yes_no_columns = [col for col in df.columns if col != 'Disorder' and col != 'age']
    for col in yes_no_columns:
        user_input[col] = st.selectbox(
            f'{col.replace("_", " ").title()}', 
            options=[None, 'Yes', 'No'],
            help=f"Indicate if you have experienced {col.replace('_', ' ')}"
        )
    
    if st.button('Predict'):
        if None in user_input.values():
            st.warning('Please fill all fields before predicting.')
        else:
            with st.spinner('Predicting...'):
                time.sleep(2)
                
                for col in yes_no_columns:
                    user_input[col] = 1 if user_input[col] == 'Yes' else 0

                input_df = pd.DataFrame([user_input])

                input_df_scaled = scaler.transform(input_df)
                prediction = model.predict(input_df_scaled)
                prediction_label = le_target.inverse_transform(prediction)

                st.session_state.prediction_label = prediction_label[0]
                st.session_state.user_input = user_input

                st.experimental_rerun()

# Function to display the result page
def display_result_page():
    st.title('Prediction Result 🎉')
    st.markdown(f"### The predicted disorder is: **{st.session_state.prediction_label}**")
    
    st.markdown("### Detailed Results")
    st.write(f"Prediction Confidence: {np.max(model.predict_proba(pd.DataFrame([st.session_state.user_input]))):.2%}")

    st.markdown("### Disorder Overview")
    disorder_overview = {
        "Depression": "Depression is a mood disorder that causes a persistent feeling of sadness and loss of interest.",
        "Anxiety": "Anxiety disorders involve more than temporary worry or fear. For people with anxiety disorders, the anxiety does not go away and can get worse over time.",
        # Add more disorder overviews as needed
    }
    
    overview = disorder_overview.get(st.session_state.prediction_label, "Overview not available for this disorder.")
    st.write(overview)
    
    if st.button('Go Back'):
        del st.session_state.prediction_label
        del st.session_state.user_input
        st.experimental_rerun()

if 'prediction_label' in st.session_state:
    display_result_page()
else:
    display_input_page()

