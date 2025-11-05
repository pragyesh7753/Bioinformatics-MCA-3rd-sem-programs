from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load genomic features dataset
data = pd.read_csv("protein_features.csv")
X = data.drop("function", axis=1)  # Features
y = data["function"]  # Target labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# What this means

# Each row represents one protein, and the columns are features describing it:

# Column	Meaning
# hydrophobicity	How water-repelling the protein is
# charge	Electric charge (+1, -1, or 0)
# molecular_weight	Weight of the molecule
# polarity	Polarity of amino acids
# function	What the protein does (target label to predict)
