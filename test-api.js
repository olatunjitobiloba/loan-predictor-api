const https = require('https');

const BASE_URL = 'loan-predictor-api-91xu.onrender.com';

// Make prediction
const data = JSON.stringify({
  ApplicantIncome: 5000,
  LoanAmount: 150,
  Credit_History: 1
});

const options = {
  hostname: BASE_URL,
  path: '/predict',
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Content-Length': data.length
  }
};

console.log('Testing production API...\n');

const req = https.request(options, (res) => {
  let body = '';
  
  res.on('data', (chunk) => {
    body += chunk;
  });
  
  res.on('end', () => {
    try {
      const result = JSON.parse(body);
      console.log('✅ API Response:');
      console.log('Prediction:', result.prediction);
      console.log('Confidence:', (result.confidence * 100).toFixed(2) + '%');
      console.log('Prediction ID:', result.prediction_id);
      console.log('\nFull Response:');
      console.log(JSON.stringify(result, null, 2));
    } catch (error) {
      console.error('❌ Error parsing response:', error);
      console.log('Raw response:', body);
    }
  });
});

req.on('error', (error) => {
  console.error('❌ Request failed:', error);
});

req.write(data);
req.end();
