import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_credit_card_fraud_model():
    # Load dataset
    df = pd.read_csv('creditcard.csv')
    
    # --- Step 2: Prepare Features and Target ---
    X = df.drop("Class", axis=1)   # TODO: all columns except target
    y = df["Class"]              # TODO: target column (0 = Not Fraud, 1 = Fraud)
    
    # --- Step 3: Scale Features to [0, 1] for MultinomialNB ---
    scaler = MinMaxScaler()                    # TODO: Initialize MinMaxScaler
    X_scaled = scaler.fit_transform(X)         # TODO: Fit and transform X
    
    # --- Step 4: Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y   # TODO: fill test size and stratify
    )
    
    # --- Step 5: Define Models ---
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),       # TODO: MultinomialNB
        "Decision Tree": DecisionTreeClassifier(random_state=42),    # TODO: DecisionTreeClassifier
        "Random Forest": RandomForestClassifier(random_state=42)     # TODO: RandomForestClassifier
    }
    
    # --- Step 6: Train and Evaluate ---
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)             # TODO: Train the model
        preds[name] = model.predict(X_test)     # TODO: Make predictions (store in dictionary)
        acc = accuracy_score(y_test, preds[name]) # TODO: Calculate accuracy
        print(f"{name} Accuracy: {acc * 100:.2f}%") 
        
    # Return all variables for testing
    return X_train, X_test, y_train, y_test, X_scaled, models, scaler

# --- Run training only when executing script directly ---
if __name__ == "__main__":
    train_credit_card_fraud_model()
