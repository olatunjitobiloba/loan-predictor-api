# 🚀 Deployment Checklist

Before deploying to Render, verify all items below are complete:

## ✅ Pre-Deployment Checklist

### 1. Code & Configuration
- [ ] All code is committed to GitHub
- [ ] `app_v4.py` is the main application file
- [ ] `Procfile` contains: `web: gunicorn app_v4:app`
- [ ] `runtime.txt` specifies Python 3.11.x
- [ ] `.env.example` documents all environment variables
- [ ] `requirements.txt` is up to date with all dependencies including `gunicorn`

### 2. Environment Variables
- [ ] `SECRET_KEY` - Generate using: `python -c "import secrets; print(secrets.token_hex(32))"`
- [ ] `DATABASE_URL` - Configured for production (PostgreSQL) or default SQLite
- [ ] `FLASK_ENV` - Set to 'production'

### 3. Application
- [ ] Model loads successfully: `models/loan_model_v2.pkl`
- [ ] Feature names loaded: `models/feature_names.txt`
- [ ] Model info loaded: `models/model_info.json`
- [ ] Database initialization works
- [ ] All endpoints are functional

### 4. Testing
- [ ] Local tests pass: `pytest`
- [ ] API responds to requests
- [ ] Health check endpoint works: `GET /health`
- [ ] Prediction endpoint works: `POST /predict`

## 🔧 Generate Secret Key

Run this command to generate a production-ready secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output and use it for `SECRET_KEY` in Render environment variables.

## 📝 Steps to Deploy on Render

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for production deployment"
git push origin main
```

### 2. Create Web Service on Render
1. Go to https://render.com
2. Sign in with GitHub
3. Click "New +" → "Web Service"
4. Select your GitHub repository
5. Fill in configuration:
   - **Name:** loan-predictor-api
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app_v4:app`

### 3. Add Environment Variables
Click "Advanced" and add:
- `SECRET_KEY` = (your generated secret key)
- `DATABASE_URL` = (optional, uses SQLite by default)
- `FLASK_ENV` = `production`

### 4. Deploy
Click "Create Web Service" and wait 3-5 minutes.

## ✅ Post-Deployment Verification

Once deployed, test these endpoints:

### Test 1: Health Check
```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "database": "connected",
  "timestamp": "2025-12-31T..."
}
```

### Test 2: API Info
```bash
curl https://your-app-name.onrender.com/api
```

### Test 3: Make Prediction
```bash
curl -X POST https://your-app-name.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Gender": "Male",
    "Married": "Yes",
    "Education": "Graduate"
  }'
```

Expected response:
```json
{
  "success": true,
  "prediction": "Approved",
  "confidence": 0.89,
  "probability": {"rejected": 0.11, "approved": 0.89},
  ...
}
```

## 🐛 Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Verify `requirements.txt` is correct
- Ensure Python 3.11 compatibility
- Check for syntax errors in all Python files

### Application Crashes
- Review runtime logs in Render dashboard
- Verify `SECRET_KEY` is set
- Check database connection string
- Ensure all model files are included in repository

### Slow First Request
- Render's free tier spins down inactive apps
- First request after inactivity takes 30-60 seconds
- Consider upgrading to paid tier for better performance

### Database Issues
- SQLite works by default (no setup needed)
- For PostgreSQL, ensure `DATABASE_URL` is set correctly
- Check database permissions and connectivity

## 📊 Monitoring

After deployment:
1. Check logs regularly in Render dashboard
2. Monitor API endpoint responses
3. Track prediction accuracy and usage
4. Review errors and warnings

## 🔗 Useful Links

- **Render Dashboard:** https://dashboard.render.com
- **Render Docs:** https://render.com/docs
- **GitHub:** https://github.com
- **Flask Documentation:** https://flask.palletsprojects.com
- **Gunicorn Documentation:** https://docs.gunicorn.org

## 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Render logs
3. Verify environment variables are set
4. Test endpoints using cURL or Postman
5. Review README.md for detailed API documentation

---

**Last Updated:** December 2025  
**Deployment Ready:** ✅ Yes
