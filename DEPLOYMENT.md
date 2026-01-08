# Deployment Guide

## Production Deployment on Render

### Prerequisites
- GitHub account with repository
- Render account (free tier available at https://render.com)
- All code committed and pushed to main branch

---

## Deployment Steps

### Step 1: Prepare Application

Ensure these files exist in your repository:

- **`requirements.txt`** - All Python dependencies (including gunicorn)
- **`Procfile`** - Deployment configuration: `web: waitress-serve --port=$PORT app_v4:app`
- **`runtime.txt`** - Python version: `python-3.11.9`
- **`.env.example`** - Environment variable template

Verify locally:
```bash
# Requirements include gunicorn
grep gunicorn requirements.txt

# Procfile is correct
cat Procfile

# Runtime is set
cat runtime.txt
```

---

### Step 2: Create Render Web Service

1. Go to https://render.com/dashboard
2. Click **"New +"** → **"Web Service"**
3. Select **"Public Git Repository"** tab
4. Enter your GitHub repository URL:
   ```
   https://github.com/yourusername/loan-predictor-api
   ```
   OR connect GitHub account and select repository

5. Configure service settings:
   - **Name:** `loan-predictor-api`
   - **Region:** US West (Oregon) or closest to you
   - **Branch:** `main`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app_v4:app`
   - **Instance Type:** `Free`

6. Click **"Create Web Service"**

---

### Step 3: Configure Environment Variables

While your service is building:

1. In Render Dashboard, go to your service → **"Environment"** tab
2. Click **"Add Environment Variable"**
3. Add the following:

#### Required Variables

**SECRET_KEY** (Generate a random 32-character string)
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Example output:
```
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0
```

**DATABASE_URL** (Optional, uses SQLite by default)
- For SQLite: `sqlite:///predictions.db`
- For PostgreSQL: See Step 4 below

**PYTHON_VERSION** (Optional)
- Value: `3.11.0`

---

### Step 4: Add PostgreSQL Database (Recommended for Production)

1. In Render Dashboard, click **"New +"** → **"PostgreSQL"**
2. Configure:
   - **Name:** `loan-predictor-db`
   - **Database:** `loan_predictions`
   - **User:** (auto-generated)
   - **Region:** Same as web service
   - **Instance Type:** `Free`
3. Click **"Create Database"**
4. Wait 2-3 minutes for provisioning
5. Copy the **Internal Database URL** (not External)
6. Go back to your Web Service → **"Environment"** tab
7. Add environment variable:
   - **Key:** `DATABASE_URL`
   - **Value:** Paste the copied internal database URL

---

### Step 5: Deploy

1. Your service should already be deploying
2. Monitor the deployment:
   - Go to **"Logs"** tab
   - Watch for "Build successful" message
   - Then "Deploying..." status
   - Finally "Successfully deployed" confirmation

3. Wait 5-10 minutes for first deployment

4. Once deployed, your API URL appears:
   ```
   https://loan-predictor-api-91xu.onrender.com
   ```

---

## Post-Deployment

### Test Deployment

Test your live API endpoints:

```bash
# Health check
curl https://loan-predictor-api-91xu.onrender.com/health

# API info
curl https://loan-predictor-api-91xu.onrender.com/api

# Make prediction
curl -X POST https://loan-predictor-api-91xu.onrender.com/predict \
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

# Get statistics
curl https://loan-predictor-api-91xu.onrender.com/statistics

# Get prediction history
curl https://loan-predictor-api-91xu.onrender.com/history
```

### Monitor Application

**Logs:**
- Render Dashboard → Your Service → **"Logs"** tab
- Check for errors and warnings
- View request logs

**Metrics:**
- Dashboard shows CPU usage, memory, response times
- Monitor for performance issues

**Database:**
- PostgreSQL dashboard shows connections and queries
- SQLite uses local file (persists in instance)

---

## Troubleshooting

### Build Fails

**Error: "pip: command not found"**
- Verify `requirements.txt` exists in repo root
- Push changes: `git add . && git commit -m "Fix requirements" && git push`
- Click **"Manual Deploy"** in Render

**Error: Missing dependencies**
- Check `requirements.txt` includes all imports
- Update locally: `pip freeze > requirements.txt`
- Commit and push

### App Crashes on Deploy

**Error: "gunicorn: command not found"**
- Ensure `gunicorn==23.0.0` is in `requirements.txt`
- Change Procfile to: `web: waitress-serve --port=$PORT app_v4:app`
- Commit and redeploy

**Error: Model not found**
- Verify `models/` folder is committed to git
- Check model files: `loan_model_v2.pkl`, `feature_names.txt`, `model_info.json`
- Use `git add models/` and push

**Error: Database connection failed**
- Verify `DATABASE_URL` is set in environment variables
- Test locally with `DATABASE_URL` value
- Check PostgreSQL instance is running (if using PostgreSQL)

### Slow First Request

**Expected behavior:**
- Free tier instances sleep after 15 minutes of inactivity
- First request takes 30-60 seconds to wake up
- Subsequent requests are normal speed (200-500ms)

**To avoid sleep:**
- Upgrade to paid tier (dedicated resources)
- Or use external monitoring service to ping periodically

### Health Check Returns 404

- Check health check path in Render: Should be `/health`
- Verify app is running: Check logs for errors
- Wait 60 seconds for service to fully start

---

## Updating Deployment

### Deploy Code Changes

```bash
# Make code changes locally
# Test locally

# Commit and push
git add .
git commit -m "Description of changes"
git push origin main

# Render automatically redeploys on every push
# Monitor logs in Render Dashboard
```

### Update Environment Variables

1. Go to service → **"Environment"** tab
2. Click variable to edit
3. Update value
4. Service automatically redeploys
5. Check logs to confirm successful deployment

---

## Custom Domain (Optional)

1. Go to service → **"Settings"** tab
2. Scroll to **"Custom Domains"**
3. Click **"Add Custom Domain"**
4. Enter your domain (e.g., `api.example.com`)
5. Follow DNS configuration instructions
6. DNS typically propagates in 24-48 hours

---

## Scaling & Upgrading

### Free Tier Limitations
- **Memory:** 512 MB RAM
- **CPU:** Shared
- **Sleep:** Hibernates after 15 minutes inactivity
- **Speed:** Cold starts take 30-60 seconds

### Upgrade to Paid Tier
1. Go to service → **"Settings"** tab
2. Find **"Instance Type"**
3. Click **"Change"** → Select paid tier
4. Service restarts with dedicated resources
5. Billing starts immediately

### Paid Tier Benefits
- ✅ No sleeping (always available)
- ✅ Dedicated CPU and memory
- ✅ Faster cold starts
- ✅ Better performance
- ✅ Professional hosting

---

## Environment Variables Reference

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask session secret | Yes | `a1b2c3d4...` |
| `DATABASE_URL` | Database connection | No | `postgresql://user:pass@host/db` |
| `PYTHON_VERSION` | Python version | No | `3.11.0` |
| `PORT` | Server port | No | `5000` (Render sets) |
| `FLASK_ENV` | Environment mode | No | `production` |

---

## Production Checklist

Before considering production-ready:

- [ ] API deployed and accessible
- [ ] Health endpoint returns 200
- [ ] Predictions work with valid data
- [ ] Validation rejects invalid data
- [ ] Database stores predictions
- [ ] Statistics endpoint works
- [ ] History endpoint returns data
- [ ] All endpoints tested in Postman
- [ ] Error handling works (400, 404, 500)
- [ ] Logs are monitoring for errors
- [ ] Database backups configured (if using PostgreSQL)
- [ ] Custom domain configured (optional)

---

## Production URL

Your API is now live at:

```
https://loan-predictor-api-91xu.onrender.com
```

Share this URL to:
- ✅ Demonstrate the project
- ✅ Include in portfolio
- ✅ Add to resume
- ✅ Show to recruiters
- ✅ Integrate with client applications

---

## Useful Links

- **Render Docs:** https://render.com/docs
- **Render Community:** https://community.render.com
- **GitHub:** [Your repository URL]
- **API Documentation:** See README.md
 - **PitchHut Project:** https://www.pitchhut.com/project/loan-predictor-api

---

## Support

For deployment issues:
1. Check Render logs first
2. Review this guide's troubleshooting section
3. Check Render documentation
4. Open GitHub issue with error details

---

**Last Updated:** December 2025
**API Version:** 4.0
**Status:** Production Ready ✅
