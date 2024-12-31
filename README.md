# Linear-Regression-Health-Costs-Calculator
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with actual dataset path)
dataset_path = "health_costs.csv"
df = pd.read_csv(dataset_path)

# Ensure the dataset has necessary columns
required_columns = ['age', 'bmi', 'children', 'smoker', 'region', 'charges']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")

# One-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['smoker', 'region'], drop_first=True)

# Define features and target variable
X = df.drop(columns=['charges'])
y = df['charges']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

def predict_health_cost(age, bmi, children, smoker, region):
    """Predict health insurance cost based on input parameters."""
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker_yes': [1 if smoker.lower() == 'yes' else 0],
        'region_northwest': [1 if region.lower() == 'northwest' else 0],
        'region_southeast': [1 if region.lower() == 'southeast' else 0],
        'region_southwest': [1 if region.lower() == 'southwest' else 0],
    })

    # Predict cost
    predicted_cost = model.predict(input_data)[0]
    return predicted_cost

# Example usage
example_cost = predict_health_cost(age=35, bmi=28.5, children=2, smoker='yes', region='southeast')
print(f"Predicted Health Insurance Cost: ${example_cost:.2f}")
