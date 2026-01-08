// API Base URL - Use relative URL for localhost, full URL for production
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? ''
    : 'https://loan-predictor-api-91xu.onrender.com';

// Load statistics and models on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStatistics();
    loadModels();
});

// Load statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_URL}/statistics`, {
            cache: 'no-store',
            headers: { 'Cache-Control': 'no-cache' },
        });
        const data = await response.json();

        document.getElementById('stats-total').textContent = data.total_predictions;
        document.getElementById('stats-approved').textContent = data.approved;
        document.getElementById('stats-rejected').textContent = data.rejected;
        document.getElementById('stats-rate').textContent = data.approval_rate;
        // Animate the hero "total-predictions" from its current displayed value
        animateNumber('total-predictions', data.total_predictions);
    } catch (error) {
        console.error('Error loading statistics:', error);
        document.getElementById('stats-total').textContent = 'Error';
    }
}

// Animate number counting (smooth, from current displayed value)
const _numberAnimations = new Map();
function animateNumber(elementId, targetNumber, duration = 800) {
    const element = document.getElementById(elementId);
    if (!element) return;

    // Parse current displayed number or default to 0
    const raw = (element.textContent || '').trim();
    const parsed = parseInt(raw.replace(/[^0-9\-]/g, ''), 10);
    const from = Number.isFinite(parsed) ? parsed : 0;
    const to = Number(targetNumber) || 0;
    if (from === to) return;

    // Cancel any existing animation for this element
    if (_numberAnimations.has(elementId)) {
        cancelAnimationFrame(_numberAnimations.get(elementId));
        _numberAnimations.delete(elementId);
    }

    const startTime = performance.now();
    const easeInOut = (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;

    function frame(now) {
        const t = Math.min(1, (now - startTime) / duration);
        const eased = easeInOut(t);
        const current = Math.round(from + (to - from) * eased);
        element.textContent = current;
        if (t < 1) {
            const raf = requestAnimationFrame(frame);
            _numberAnimations.set(elementId, raf);
        } else {
            element.textContent = to;
            _numberAnimations.delete(elementId);
        }
    }

    const raf = requestAnimationFrame(frame);
    _numberAnimations.set(elementId, raf);
}

// Load and display available models
async function loadModels() {
    try {
        const response = await fetch(`${API_URL}/models`);
        const data = await response.json();

        const modelsGrid = document.getElementById('models-grid');
        if (!modelsGrid) return; // Exit if element doesn't exist

        modelsGrid.innerHTML = '';

        const bestModel = data.best_model;
        // Determine rendering order. Prefer ordered list from server (`models_list`),
        // then `available_models`, then the object keys as a last resort.
        const order = Array.isArray(data.models_list)
            ? data.models_list.map(m => m.id)
            : (Array.isArray(data.available_models) ? data.available_models : Object.keys(data.models || {}));

        order.forEach(modelKey => {
            // Get model data from the ordered list if present, otherwise from data.models
            const modelData = Array.isArray(data.models_list)
                ? (data.models_list.find(m => m.id === modelKey) || {})
                : (data.models && data.models[modelKey] ? data.models[modelKey] : {});

            const isBest = modelKey === bestModel;

            // Safely format metrics (handle missing/null values)
            const acc = (typeof modelData.accuracy === 'number') ? (modelData.accuracy * 100).toFixed(2) + '%' : 'N/A';
            const prec = (typeof modelData.precision === 'number') ? (modelData.precision * 100).toFixed(2) + '%' : 'N/A';
            const rec = (typeof modelData.recall === 'number') ? (modelData.recall * 100).toFixed(2) + '%' : 'N/A';
            const f1 = (typeof modelData.f1_score === 'number') ? (modelData.f1_score * 100).toFixed(2) + '%' : 'N/A';
            const predTime = (typeof modelData.avg_prediction_time === 'number') ? (modelData.avg_prediction_time * 1000).toFixed(2) + 'ms' : 'N/A';

            const card = document.createElement('div');
            card.className = `model-card ${isBest ? 'best' : ''}`;
            card.innerHTML = `
                <div class="model-header">
                    <h3 class="model-name">${modelData.name || modelKey}</h3>
                    ${isBest ? '<span class="best-badge">🏆 BEST</span>' : ''}
                </div>
                <div class="model-metrics">
                    <div class="metric-row">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">${acc}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Precision</span>
                        <span class="metric-value">${prec}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Recall</span>
                        <span class="metric-value">${rec}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">F1-Score</span>
                        <span class="metric-value">${f1}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-label">Prediction Time</span>
                        <span class="metric-value">${predTime}</span>
                    </div>
                </div>
            `;

            modelsGrid.appendChild(card);
        });
        // Also populate the performance metrics table (best model + summary)
        if (typeof renderMetricsTable === 'function') {
            renderMetricsTable(data.models, data.best_model);
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

// Render the metrics table under the Model Performance Metrics section
function renderMetricsTable(models, bestModelKey) {
    const tbody = document.getElementById('metrics-tbody');
    if (!tbody) return;

    // Clear existing rows
    tbody.innerHTML = '';

    if (!models || Object.keys(models).length === 0) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="6" class="loading-cell">No metrics available</td>';
        tbody.appendChild(tr);
        return;
    }

    // Determine best model object
    let bestModel = models[bestModelKey];
    if (!bestModel) {
        // Fallback: choose model with highest accuracy
        bestModel = Object.values(models).reduce((best, m) => {
            if (!best) return m;
            const a = typeof m.accuracy === 'number' ? m.accuracy : -Infinity;
            const b = typeof best.accuracy === 'number' ? best.accuracy : -Infinity;
            return a > b ? m : best;
        }, null);
    }

    // Compute averages
    let sumAcc = 0, sumPrec = 0, sumRec = 0, sumF1 = 0;
    let countAcc = 0, countPrec = 0, countRec = 0, countF1 = 0;
    Object.values(models).forEach(m => {
        if (typeof m.accuracy === 'number') { sumAcc += m.accuracy; countAcc++; }
        if (typeof m.precision === 'number') { sumPrec += m.precision; countPrec++; }
        if (typeof m.recall === 'number') { sumRec += m.recall; countRec++; }
        if (typeof m.f1_score === 'number') { sumF1 += m.f1_score; countF1++; }
    });

    const avgAcc = countAcc ? (sumAcc / countAcc) : null;
    const avgPrec = countPrec ? (sumPrec / countPrec) : null;
    const avgRec = countRec ? (sumRec / countRec) : null;
    const avgF1 = countF1 ? (sumF1 / countF1) : null;

    // Best model row
    if (bestModel) {
        const trBest = document.createElement('tr');
        const name = bestModel.name || bestModelKey || 'Best Model';
        const accuracy = typeof bestModel.accuracy === 'number' ? (bestModel.accuracy * 100).toFixed(2) + '%' : 'N/A';
        const precision = typeof bestModel.precision === 'number' ? (bestModel.precision * 100).toFixed(2) + '%' : 'N/A';
        const recall = typeof bestModel.recall === 'number' ? (bestModel.recall * 100).toFixed(2) + '%' : 'N/A';
        const f1 = typeof bestModel.f1_score === 'number' ? (bestModel.f1_score * 100).toFixed(2) + '%' : 'N/A';
        const status = bestModel.loaded ? 'Best (Loaded)' : 'Best (Not loaded)';

        trBest.innerHTML = `
            <td>${name}</td>
            <td>${accuracy}</td>
            <td>${precision}</td>
            <td>${recall}</td>
            <td>${f1}</td>
            <td>${status}</td>
        `;
        tbody.appendChild(trBest);
    }

    // Summary / averages row
    const trAvg = document.createElement('tr');
    trAvg.innerHTML = `
        <td><strong>Average</strong></td>
        <td><strong>${avgAcc !== null ? (avgAcc * 100).toFixed(2) + '%' : 'N/A'}</strong></td>
        <td><strong>${avgPrec !== null ? (avgPrec * 100).toFixed(2) + '%' : 'N/A'}</strong></td>
        <td><strong>${avgRec !== null ? (avgRec * 100).toFixed(2) + '%' : 'N/A'}</strong></td>
        <td><strong>${avgF1 !== null ? (avgF1 * 100).toFixed(2) + '%' : 'N/A'}</strong></td>
        <td><strong>Summary</strong></td>
    `;
    tbody.appendChild(trAvg);
}

// Benchmark all models
async function benchmarkModels() {
    const formData = new FormData(document.getElementById('prediction-form'));
    const data = {};

    formData.forEach((value, key) => {
        if (['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'].includes(key)) {
            data[key] = value ? parseFloat(value) : undefined;
        } else {
            data[key] = value;
        }
    });

    // Check if form has required data
    if (!data.ApplicantIncome) {
        alert('Please fill in at least the Applicant Income field');
        return;
    }

    // Show loading state
    const benchmarkBtn = document.getElementById('benchmark-btn');
    if (benchmarkBtn) {
        benchmarkBtn.disabled = true;
        benchmarkBtn.textContent = 'Running Benchmark...';
    }

    try {
        const response = await fetch(`${API_URL}/models/benchmark`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (response.ok) {
            displayBenchmarkResults(result);
        } else {
            alert('Error: ' + (result.error || 'Benchmark failed'));
        }
    } catch (error) {
        console.error('Benchmark error:', error);
        alert('Error running benchmark. Please ensure the form is filled correctly.');
    } finally {
        // Reset button
        if (benchmarkBtn) {
            benchmarkBtn.disabled = false;
            benchmarkBtn.textContent = 'Compare All Models';
        }
    }
}

// Display benchmark results
function displayBenchmarkResults(result) {
    const container = document.getElementById('benchmark-results');
    if (!container) return;

    container.style.display = 'block';

    let html = '<h3 style="margin-bottom: 1.5rem; color: var(--primary-color);">Benchmark Results</h3>';

    // Consensus
    const consensusColor = result.consensus.prediction === 'Approved' ? 'var(--success-color)' : 'var(--danger-color)';
    html += `
        <div class="consensus-box" style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border: 2px solid ${consensusColor};">
            <div style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.5rem;">Model Consensus</div>
            <div class="consensus-prediction" style="font-size: 2rem; font-weight: 700; color: ${consensusColor}; margin-bottom: 0.5rem;">
                ${result.consensus.prediction}
            </div>
            <div class="consensus-agreement" style="font-size: 0.875rem; color: var(--text-secondary);">
                ${result.consensus.agreement} agreement (${result.consensus.models_agree}/${result.consensus.total_models} models)
            </div>
        </div>
    `;

    // Individual results
    html += '<div class="benchmark-results-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">';
    Object.entries(result.results).forEach(([modelName, modelResult]) => {
        const isApproved = modelResult.prediction === 'Approved';
        const borderColor = isApproved ? 'var(--success-color)' : 'var(--danger-color)';
        const predictionColor = isApproved ? 'var(--success-color)' : 'var(--danger-color)';

        html += `
            <div class="benchmark-result-card" style="background: var(--card-bg); padding: 1.5rem; border-radius: 12px; border: 2px solid ${borderColor}; transition: transform 0.3s ease;">
                <h4 style="margin: 0 0 1rem 0; font-size: 1rem; color: var(--text-primary);">
                    ${modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </h4>
                <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: ${predictionColor};">
                    ${modelResult.prediction}
                </div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    Confidence: ${(modelResult.confidence * 100).toFixed(1)}%
                </div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    Time: ${modelResult.prediction_time}
                </div>
            </div>
        `;
    });
    html += '</div>';

    container.innerHTML = html;
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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

    // Hide benchmark results if visible
    const benchmarkResults = document.getElementById('benchmark-results');
    if (benchmarkResults) {
        benchmarkResults.style.display = 'none';
    }

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
