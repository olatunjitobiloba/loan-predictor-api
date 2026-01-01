// API Base URL - Use relative URL for localhost, full URL for production
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? '' 
    : 'https://loan-predictor-api-91xu.onrender.com';

// Load statistics on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStatistics();
    loadTotalPredictions();
});

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_URL}/statistics`);
        const data = await response.json();
        
        document.getElementById('stats-total').textContent = data.total_predictions;
        document.getElementById('stats-approved').textContent = data.approved;
        document.getElementById('stats-rejected').textContent = data.rejected;
        document.getElementById('stats-rate').textContent = data.approval_rate;
        document.getElementById('total-predictions').textContent = data.total_predictions;
    } catch (error) {
        console.error('Error loading statistics:', error);
        document.getElementById('stats-total').textContent = 'Error';
    }
}

// Load total predictions for hero
async function loadTotalPredictions() {
    try {
        const response = await fetch(`${API_URL}/statistics`);
        const data = await response.json();
        animateNumber('total-predictions', data.total_predictions);
    } catch (error) {
        console.error('Error loading total predictions:', error);
    }
}

// Animate number counting
function animateNumber(elementId, targetNumber) {
    const element = document.getElementById(elementId);
    let current = 0;
    const increment = targetNumber / 50;
    const timer = setInterval(() => {
        current += increment;
        if (current >= targetNumber) {
            element.textContent = targetNumber;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current);
        }
    }, 20);
}

// Handle form submission
document.getElementById('prediction-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(e.target);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert numeric fields
        if (['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'].includes(key)) {
            data[key] = value ? parseFloat(value) : undefined;
        } else {
            data[key] = value;
        }
    });
    
    // Show loading state
    const submitBtn = document.getElementById('submit-btn');
    const submitText = document.getElementById('submit-text');
    const submitLoader = document.getElementById('submit-loader');
    
    submitBtn.disabled = true;
    submitText.style.display = 'none';
    submitLoader.style.display = 'block';
    
    try {
        // Make prediction
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Show results
            displayResults(result);
            
            // Reload statistics
            loadStatistics();
        } else {
            // Show error
            alert('Error: ' + (result.validation_errors ? result.validation_errors.join(', ') : result.error));
        }
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error connecting to API. Please try again.');
    } finally {
        // Reset button
        submitBtn.disabled = false;
        submitText.style.display = 'block';
        submitLoader.style.display = 'none';
    }
});

// Display results
function displayResults(result) {
    const resultsContainer = document.getElementById('results-container');
    const resultCard = document.getElementById('result-card');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultSubtitle = document.getElementById('result-subtitle');
    
    // Show results container
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Set result based on prediction
    const isApproved = result.prediction === 'Approved';
    
    if (isApproved) {
        resultIcon.textContent = '✓';
        resultIcon.classList.remove('rejected');
        resultTitle.textContent = 'Loan Approved!';
        resultTitle.style.color = 'var(--success-color)';
        resultSubtitle.textContent = 'Congratulations! Your loan application is likely to be approved.';
    } else {
        resultIcon.textContent = '✗';
        resultIcon.classList.add('rejected');
        resultTitle.textContent = 'Loan Rejected';
        resultTitle.style.color = 'var(--danger-color)';
        resultSubtitle.textContent = 'Unfortunately, your loan application is likely to be rejected.';
    }
    
    // Set confidence
    const confidence = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence-value').textContent = confidence + '%';
    document.getElementById('confidence-fill').style.width = confidence + '%';
    
    // Set probabilities
    const approvalProb = (result.probability.approved * 100).toFixed(1);
    const rejectionProb = (result.probability.rejected * 100).toFixed(1);
    
    document.getElementById('approval-prob').textContent = approvalProb + '%';
    document.getElementById('approval-fill').style.width = approvalProb + '%';
    
    document.getElementById('rejection-prob').textContent = rejectionProb + '%';
    document.getElementById('rejection-fill').style.width = rejectionProb + '%';
    
    // Show warnings if any
    const warningsContainer = document.getElementById('warnings-container');
    const warningsList = document.getElementById('warnings-list');
    
    if (result.warnings && result.warnings.length > 0) {
        warningsContainer.style.display = 'block';
        warningsList.innerHTML = '';
        result.warnings.forEach(warning => {
            const li = document.createElement('li');
            li.textContent = warning;
            warningsList.appendChild(li);
        });
    } else {
        warningsContainer.style.display = 'none';
    }
}

// Reset form
function resetForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('results-container').style.display = 'none';
    document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
