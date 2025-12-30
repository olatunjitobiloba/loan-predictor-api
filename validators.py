"""
Input validation for Loan Prediction API
"""

from typing import Dict, List, Tuple, Any

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class LoanApplicationValidator:
    """Validates loan application data"""

    # Define valid values for categorical fields
    VALID_GENDER = ['Male', 'Female']
    VALID_MARRIED = ['Yes', 'No']
    VALID_DEPENDENTS = ['0', '1', '2', '3+']
    VALID_EDUCATION = ['Graduate', 'Not Graduate']
    VALID_SELF_EMPLOYED = ['Yes', 'No']
    VALID_PROPERTY_AREA = ['Urban', 'Semiurban', 'Rural']

    # Define valid values for categorical fields
    VALID_GENDER = ['Male', 'Female']
    VALID_MARRIED = ['Yes', 'No']
    VALID_DEPENDENTS = ['0', '1', '2', '3+']
    VALID_EDUCATION = ['Graduate', 'Not Graduate']
    VALID_SELF_EMPLOYED = ['Yes', 'No']
    VALID_PROPERTY_AREA = ['Urban', 'Semiurban', 'Rural']

    # Define reasonable ranges
    MIN_INCOME = 0
    MAX_INCOME = 100000 # 100k per month
    MIN_LOAN_AMOUNT = 0
    MAX_LOAN_AMOUNT = 10000 # 10M (in thousands)
    VALID_LOAN_TERMS = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]

    @staticmethod
    def validate_required_fields(data: Dict) -> Tuple[bool, str]:
        """
        Check if required fields are present

        Returns:
            (is_valid, error_message)
        """
        required = ['ApplicantIncome']

        missing = [field for field in required
                   if field not in data or data[field] is None]

        if missing:
            return False, f"Missing required fields: {','.join(missing)}"
        
        return True, ""
    
    @staticmethod
    def validate_numeric_field(data: Dict, field: str, min_val: float = None,
                               max_val: float = None, required: bool = False) -> Tuple[bool, str]:
        """
        Validate a numeric field

        Returns:
            (is_valid, error_message)
        """
        # Check if field exists
        if field not in data:
            if required:
                return False, f"{field} is required"
            return True, "" # Optional field, not present

        value = data[field]

        # Check if it's a number
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
                data[field] = value # Convert to float
            except (ValueError, TypeError):
                return False, f"{field} must be a number"
        
        # Check if it's positive (for most financial fields)
        if value < 0:
            return False, f"{field} cannot be negative"
        
        # Check min value
        if min_val is not None and value < min_val:
            return False, f"{field} must be at least {min_val}"
        
        # Check max value
        if max_val is not None and value > max_val:
            return False, f"{field} cannot exceed {max_val}"
        
        return True, ""
    
    @staticmethod
    def validate_categorical_field(data: Dict, field: str, valid_values:
                                   List[str],
                                   required: bool = False) -> Tuple[bool,
                                                                    str]:
        """
        Validate a categorical field

        Returns:
            (is_valid, error_message)
        """
        # Check if field exists
        if field not in data:
            if required:
                return False, f"{field} is required"
            return True, "" # Optional field
        
        value = data[field]

        # Check if value is valid
        if value not in valid_values:
            return False, f"{field} must be one of: {','.join(valid_values)}"
        
        return True, ""
    
    @classmethod
    def validate_loan_application(cls, data: Dict) -> Tuple[bool,
                                                            List[str], List[str]]:
        """
        Comprehensive validation of loan application data

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # 1. Check required fields
        is_valid, error = cls.validate_required_fields(data)
        if not is_valid:
            errors.append(error)
            return False, errors, warnings
        
        # 2. Validate numeric fields
        numeric_validations = [
            ('ApplicantIncome', cls.MIN_INCOME, cls.MAX_INCOME, True),
            ('CoapplicantIncome', cls.MIN_INCOME, cls.MAX_INCOME, False),
            ('LoanAmount', cls.MIN_LOAN_AMOUNT, cls.MAX_LOAN_AMOUNT, False),
            ('Credit_History', 0, 1, False),
        ]

        for field, min_val, max_val, required in numeric_validations:
            is_valid, error = cls.validate_numeric_field(data, field,
                                                         min_val, max_val, required)
            if not is_valid:
                errors.append(error)
        
        # 3. Validate Loan_Amount_Term specially
        if 'Loan_Amount_Term' in data:
            is_valid, error = cls.validate_numeric_field(data, 'Loan_Amount_Term')
            if not is_valid:
                errors.append(error)
            elif data['Loan_Amount_Term'] not in cls.VALID_LOAN_TERMS:
                warnings.append(
                    f"Loan_Amount_Term {data['Loan_Amount_Term']} is unusual. "
                    f"Common values: {cls.VALID_LOAN_TERMS}"
                )
        
        # 4. Validate categorical fields
        categorical_validations = [
            ('Gender', cls.VALID_GENDER, False),
            ('Married', cls.VALID_MARRIED, False),
            ('Dependents', cls.VALID_DEPENDENTS, False),
            ('Education', cls.VALID_EDUCATION, False),
            ('Self_Employed', cls.VALID_SELF_EMPLOYED, False),
            ('Property_Area', cls.VALID_PROPERTY_AREA, False),
        ]

        for field, valid_values, required in categorical_validations:
            is_valid, error = cls.validate_categorical_field(data, field,
valid_values, required)
            if not is_valid:
                errors.append(error)

        # If any hard validation errors occurred, stop before business warnings
        if errors:
            return False, errors, warnings
        
        # 5. Business logic validations (warnings, not errors)

        # Check if loan amount is reasonable relative to income
        if 'LoanAmount' in data and 'ApplicantIncome' in data:
            total_income = data['ApplicantIncome']
            if 'CoapplicantIncome' in data:
                total_income += data['CoapplicantIncome']

            # LoanAmount is in thousands, income is monthly
            # Annual income = monthly * 12
            annual_income = total_income * 12
            loan_amount_actual = data['LoanAmount'] * 1000

            if loan_amount_actual > annual_income * 5:
                warnings.append(
                    "Loan amount is more than 5x annual income. "
                    "This may affect approval chances."
                )
        
        # Check if applicant has very low income
        if 'ApplicantIncome' in data and data['ApplicantIncome'] < 1000:
            warnings.append(
                "Applicant income is very low. This may affect approval chances."
            )
        
        # Check credit history
        if 'Credit_History' in data and data['Credit_History'] == 0:
            warnings.append(
                "No credit history. This significantly reduces approval chances."
            )

        # Return validation results
        is_valid = len(errors) == 0
        return is_valid, errors, warnings


# Test the validator
if __name__ == '__main__':
    validator = LoanApplicationValidator()

    # Test case 1: Valid data
    print("="*60)
    print("TEST 1: Valid data")
    print("="*60)
    test_data = {
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 1500,
        'LoanAmount': 150,
        'Credit_History': 1,
        'Gender': 'Male',
        'Married': 'Yes'
    }
    is_valid, errors, warnings = validator.validate_loan_application(test_data)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")

    # Test case 2: Missing required field
    print("\n" + "="*60)
    print("TEST 2: Missing required field")
    print("="*60)
    test_data = {
        'LoanAmount': 150
    }
    is_valid, errors, warnings = validator.validate_loan_application(test_data)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")

    # Test case 3: Invalid data types
    print("\n" + "="*60)
    print("TEST 3: Invalid data types")
    print("="*60)
    test_data = {
        'ApplicantIncome': 'not a number',
        'Gender': 'Other'
    }
    is_valid, errors, warnings = validator.validate_loan_application(test_data)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")

    # Test case 4: Out of range
    print("\n" + "="*60)
    print("TEST 4: Out of range values")
    print("="*60)
    test_data = {
        'ApplicantIncome': -1000,
        'LoanAmount': 50000
    }
    is_valid, errors, warnings = validator.validate_loan_application(test_data)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")

    # Test case 5: Business logic warnings
    print("\n" + "="*60)
    print("TEST 5: Business logic warnings")
    print("="*60)
    test_data = {
        'ApplicantIncome': 2000,
        'LoanAmount': 500,
        'Credit_History': 0
    }
    is_valid, errors, warnings = validator.validate_loan_application(test_data)
    print(f"Valid: {is_valid}")
    print(f"Errors: {errors}")
    print(f"Warnings: {warnings}")
            