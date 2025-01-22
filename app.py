import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import altair as alt


try:
    # Load the data
    df = pd.read_csv("software_project_data.csv")
except FileNotFoundError:
    st.write("Error: Data file not found.")
    st.stop()

# Prepare the data for the model
X = df[['Project_Size', 'Num_Developers', 'Complexity']]
y = df['Completion_Time']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create a Streamlit app
st.title("Software Project Completion Time Estimator")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Visualization", "Model Performance", "Prediction"])

# Tab 1: Data Visualization
with tab1:
    st.write("### Software Project Data")
    st.write(df)

    # Scatter plot
    st.write("### Scatter Plot of Project Variables vs Completion Time")
    for col in ['Project_Size', 'Num_Developers', 'Complexity']:
        st.write(f"**{col} vs Completion Time**")
        st.altair_chart(
            alt.Chart(df).mark_circle().encode(
                x=col,
                y='Completion_Time',
                tooltip=[col, 'Completion_Time']
            ).interactive(),
            use_container_width=True
        )

# Tab 2: Model Performance
with tab2:
    st.write("### Model Performance")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")

# Tab 3: Prediction
with tab3:
    st.write("### Predict Completion Time")
    project_size_input = st.number_input("Project Size (Lines of Code)", min_value=1000, value=10000, step=1000)
    num_developers_input = st.number_input("Number of Developers", min_value=1, value=5, step=1)
    complexity_input = st.slider("Complexity (1-5)", min_value=1, max_value=5, value=3, step=1)

    if st.button("Predict"):
        # Create input array for prediction
        input_data = [[project_size_input, num_developers_input, complexity_input]]

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.write(f"### Predicted Completion Time: {prediction:.2f}")