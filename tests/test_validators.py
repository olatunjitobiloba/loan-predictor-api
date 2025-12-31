"""
Unit tests for validation module
"""

import pytest
from validators import LoanApplicationValidator

class TestLoanApplicationValidator:
    """Test suite for LoanApplicationValidator"""

    def setup_method(self):
        """Set up test fixtures"""
        self.validator = LoanApplicationValidator()
    
    # Test required fields validation
    def test_validate_required_fields_present(self):
        """Test that validation passes when required fields are
present"""
        data = {"ApplicantIncome": 5000}
        is_valid, error = self.validator.validate_required_fields(data)
        assert is_valid == True
        assert error == ""
    
    def test_validate_required_fields_missing(self):
        """Test that validation fails when required fields are missing"""
        data = {"LoanAmount": 150}
        is_valid, error = self.validator.validate_required_fields(data)
        assert is_valid == False
        assert "ApplicantIncome" in error
    
    # Test numeric field validation
    def test_validate_numeric_field_valid(self):
        """Test numeric validation with valid data"""
        data = {"ApplicantIncome": 5000}
        is_valid, error = self.validator.validate_numeric_field(
            data, "ApplicantIncome", min_val=0, max_val=100000,
required=True
        )
        assert is_valid == True
        assert error == ""
    
    def test_validate_numeric_field_negative(self):
        """Test that negative values are rejected"""
        data = {"ApplicantIncome": -1000}
        is_valid, error = self.validator.validate_numeric_field(
            data, "ApplicantIncome", min_val=0
        )
        assert is_valid == False
        assert "cannot be negative" in error

    def test_validate_numeric_field_exceeds_max(self):
        """Test that values exceeding max are rejected"""
        data = {"ApplicantIncome": 200000}
        is_valid, error = self.validator.validate_numeric_field(
            data, "ApplicantIncome", max_val = 100000
        )
        assert is_valid == False
        assert "cannot exceed" in error
    
    def test_validate_numeric_field_not_a_number(self):
        """Test that non-numeric values are rejected"""
        data = {"ApplicantIncome": "not a number"}
        is_valid, error = self.validator.validate_numeric_field(
            data, "ApplicantIncome"
        )
        assert is_valid == False
        assert "must be a number" in error

    def test_validate_numeric_field_optional_missing(self):
        """Test that optional missing fields are valid"""
        data = {}
        is_valid, error = self.validator.validate_numeric_field(
            data, "CoapplicantIncome", required=False
        )
        assert is_valid == True

    # Test categorical field validation
    def test_validate_categorical_field_valid(self):
        """Test categorical validation with valid value"""
        data = {"Gender": "Male"}
        is_valid, error = self.validator.validate_categorical_field(
            data, "Gender", ["Male", "Female"]
        )
        assert is_valid == True
        assert error == ""
    
    def test_validate_categorical_field_invalid(self):
        """Test that invalid categorical values arre rejected"""
        data = {"Gender": "Other"}
        is_valid, error = self.validator.validate_categorical_field(
            data, "Gender", ["Male", "Female"]
        )
        assert is_valid == False
        assert "must be one of" in error
    
    def test_validate_categorical_field_optional_missing(self):
        """Test that optional missing categorical fields are valid"""
        data = {}
        is_valid, error = self.validator.validate_categorical_field(
            data, "Gender", ["Male", "Female"], required=False
        )
        assert is_valid == True
    
    # Test full loan application validation
    def test_validate_loan_application_valid_complete(self):
        """Test validation with complete valid data"""
        data = {
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 1500,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1,
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "Property_Area": "Urban"
        }
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_loan_application_valid_minimal(self):
        """Test validation with minimal valid data"""
        data = {"ApplicantIncome": 5000}
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_loan_application_missing_required(self):
        """Test validation fails without required fields"""
        data = {"LoanAmount": 150}
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == False
        assert len(errors) > 0
        assert any("ApplicantIncome" in error for error in errors)
    
    def test_validate_loan_application_multiple_errors(self):
        """Test validation catches multiple errors"""
        data = {
            "ApplicantIncome": -1000, # Negative
            "Gender": "Other", # Invalid
            "LoanAmount": 50000 # Too high
        }
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == False
        assert len(errors) >=2

    def test_validate_loan_application_warnings_low_income(self):
        """Test that low income generates warning"""
        data = {
            "ApplicantIncome": 500,
            "Credit_History": 1
        }
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == True
        assert len(warnings) > 0
        assert any("low" in warning.lower() for warning in warnings)
    
    def test_validate_loan_application_warnings_no_credit(self):
        """Test that no credit history generate warning"""
        data = {
            "ApplicantIncome": 5000,
            "Credit_History": 0
        }
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == True
        assert len(warnings) > 0
        assert any("credit" in warning.lower() for warning in warnings)
    
    def test_validate_loan_application_warnings_high_loan_ratio(self):
        """Test that high loan to income ratio generates warning"""
        data = {
            "ApplicantIncome": 2000,
            "LoanAmount": 500 # Very high relative to income
        }
        is_valid, errors, warnings = self.validator.validate_loan_application(data)
        assert is_valid == True
        assert len(warnings) > 0
    
# Run tests if this file is executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])