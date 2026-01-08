import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class LoanDataPreprocessor:
    """
    Comprehensive preprocessing pipeline for loan prediction data.
    Handles both training and test data with proper encoding and feature engineering.
    """

    def __init__(self):
        self.label_encoders = {}
        self.fill_values = {}
        self.is_fitted = False

    def fit(self, df):
        """
        Fit the preprocessor on training data.
        Learns encoding mappings and fill values.

        Parameters:
        - df: Training dataframe

        Returns:
        - self
        """
        print("=" * 60)
        print("FITTING PREPROCESSOR ON TRAINING DATA")
        print("=" * 60)

        # Store fill values for numeric columns
        numeric_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
        for col in numeric_cols:
            if col in df.columns:
                self.fill_values[col] = df[col].median()
                print(f"✓ {col} median: {self.fill_values[col]}")

        # Store mode for categorical columns
        categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
        for col in categorical_cols:
            if col in df.columns:
                self.fill_values[col] = (
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                )
                print(f"✓ {col} mode: {self.fill_values[col]}")

        # Fit label encoders for categorical columns
        print("\n" + "=" * 60)
        print("FITTING LABEL ENCODERS")
        print("=" * 60)

        encode_cols = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]

        for col in encode_cols:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                # Fill missing values before fitting
                temp_data = df[col].fillna(self.fill_values.get(col, "Unknown"))
                self.label_encoders[col].fit(temp_data)
                print(f"✓ {col}: {list(self.label_encoders[col].classes_)}")

        self.is_fitted = True
        print("\n✅ Preprocessor fitted successfully!")
        return self

    def transform(self, df, is_training=True):
        """
        Transform the dataframe using fitted preprocessor.

        Parameters:
        - df: Input dataframe
        - is_training: If True, includes target variable processing

        Returns:
        - Preprocessed dataframe
        """
        if not self.is_fitted:
            raise ValueError(
                "Preprocessor must be fitted before transform. Call fit() first."
            )

        df = df.copy()

        print("\n" + "=" * 60)
        print(f"TRANSFORMING {'TRAINING' if is_training else 'TEST'} DATA")
        print("=" * 60)
        print(f"Input shape: {df.shape}")

        # 1. Handle Missing Values
        print("\n1. Handling missing values...")

        # Numeric columns
        numeric_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
        for col in numeric_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col].fillna(self.fill_values[col], inplace=True)
                    print(f"   ✓ {col}: Filled {missing_count} missing values")
                    print(f"     with: {self.fill_values[col]}")

        # Categorical columns
        categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
        for col in categorical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col].fillna(self.fill_values[col], inplace=True)
                    print(f"   ✓ {col}: Filled {missing_count} missing values")
                    print(f"     with: '{self.fill_values[col]}'")

        # 2. Feature Engineering
        print("\n2. Creating new features...")

        # Total Income
        df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        print("   ✓ TotalIncome = ApplicantIncome + CoapplicantIncome")

        # Income to Loan Ratio
        df["Income_Loan_Ratio"] = df["TotalIncome"] / (
            df["LoanAmount"] + 1
        )  # +1 to avoid division by zero
        print("   ✓ Income_Loan_Ratio = TotalIncome / LoanAmount")

        # Loan Amount per Term
        df["Loan_Amount_Per_Term"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)
        print("   ✓ Loan_Amount_Per_Term = LoanAmount / Loan_Amount_Term")

        # EMI (Estimated Monthly Installment)
        df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] / 12 + 1)
        print("   ✓ EMI = LoanAmount / (Loan_Amount_Term / 12)")

        # Balance Income (Income after EMI)
        df["Balance_Income"] = df["TotalIncome"] - (df["EMI"] * 1000)
        print("   ✓ Balance_Income = TotalIncome - (EMI * 1000)")

        # Log transformations for skewed features
        df["Log_ApplicantIncome"] = np.log1p(df["ApplicantIncome"])
        df["Log_CoapplicantIncome"] = np.log1p(df["CoapplicantIncome"])
        df["Log_LoanAmount"] = np.log1p(df["LoanAmount"])
        df["Log_TotalIncome"] = np.log1p(df["TotalIncome"])
        print("   ✓ Log transformations applied to income and loan features")

        # 3. Encode Categorical Variables
        print("\n3. Encoding categorical variables...")

        encode_cols = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Property_Area",
        ]

        for col in encode_cols:
            if col in df.columns and col in self.label_encoders:
                # Handle unseen categories
                le = self.label_encoders[col]

                # Create a mapping with default value for unseen categories
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                default_value = 0  # Default to first class

                df[col + "_Encoded"] = df[col].map(
                    lambda x: mapping.get(x, default_value)
                )
                print(f"   ✓ {col} encoded to {col}_Encoded")

        # 4. Handle Target Variable (only for training)
        if is_training and "Loan_Status" in df.columns:
            print("\n4. Encoding target variable...")
            df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})
            print("   ✓ Loan_Status: Y=1, N=0")

        # 5. Drop original categorical columns and ID
        print("\n5. Dropping original categorical columns...")
        cols_to_drop = ["Loan_ID"] + encode_cols
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df.drop(cols_to_drop, axis=1, inplace=True)
        print(f"   ✓ Dropped: {cols_to_drop}")

        print("\n✅ Transformation complete!")
        print("Output shape:", df.shape)
        print("Final columns:", df.columns.tolist())

        return df

    def fit_transform(self, df, is_training=True):
        """
        Fit and transform in one step.

        Parameters:
        - df: Input dataframe
        - is_training: If True, includes target variable processing

        Returns:
        - Preprocessed dataframe
        """
        self.fit(df)
        return self.transform(df, is_training)


def preprocess_loan_data(train_df, test_df=None):
    """
    Convenience function to preprocess both train and test data.

    Parameters:
    - train_df: Training dataframe
    - test_df: Test dataframe (optional)

    Returns:
    - If test_df is None: (preprocessed train_df, preprocessor)
    - If test_df is provided: (preprocessed train_df, preprocessed test_df, preprocessor)
    """
    preprocessor = LoanDataPreprocessor()

    # Fit and transform training data
    train_processed = preprocessor.fit_transform(train_df, is_training=True)

    # Transform test data if provided
    if test_df is not None:
        test_processed = preprocessor.transform(test_df, is_training=False)
        return train_processed, test_processed, preprocessor

    return train_processed, preprocessor


# Example usage (commented out - uncomment to test)
if __name__ == "__main__":
    # Load your data
    train = pd.read_csv("data/train_u6lujuX_CVtuZ9i.csv")
    test = pd.read_csv("data/test_Y3wMUE5_7gLdaTN.csv")

    # Preprocess both datasets
    train_processed, test_processed, preprocessor = preprocess_loan_data(train, test)

    # Now ready for modeling!
    X_train = train_processed.drop("Loan_Status", axis=1)
    y_train = train_processed["Loan_Status"]
    X_test = test_processed  # No target variable in test

    print("\n" + "=" * 60)
    print("READY FOR MODELING")
    print("=" * 60)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
