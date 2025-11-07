# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from Bio import SeqIO

# Load gene expression data
# X: gene expression matrix (samples × genes)
# y: disease labels (0=healthy, 1=disease)
data = pd.read_csv("gene_expression_1.csv")
X = data.iloc[:, :-1]  # All columns except last
y = data.iloc[:, -1]  # Last column (labels)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize Random Forest with 100 trees
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
print(classification_report(y_test, y_pred))

# Feature importance for top genes
importances = rf_model.feature_importances_
top_genes = X.columns[importances.argsort()[-10:][::-1]]
print("Top 10 important genes:", list(top_genes))

print("------------------")

# record = SeqIO.read("gene_expression_1.csv", "csv")
# print(len(record.seq))
print("Number of samples:", len(data))
print("Number of features (genes):", len(data.columns) - 1)
