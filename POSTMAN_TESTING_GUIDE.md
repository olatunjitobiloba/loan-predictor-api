# Postman Testing Guide for Loan Predictor API

## Fixing the 415 Error

The 415 error happens when Postman doesn't send the correct `Content-Type` header. Here's how to fix it:

## Step-by-Step: Testing POST /predict

### Test 1: Basic POST /predict Request

1. **Open Postman** and create a new request
2. **Set Method**: Select `POST` from the dropdown
3. **Set URL**: `http://localhost:5000/predict`
4. **Set Headers**:
   - Click on the **"Headers"** tab
   - Add a new header:
     - **Key**: `Content-Type`
     - **Value**: `application/json`
   - (Postman might auto-add this when you select JSON body, but verify it's there)

5. **Set Body**:
   - Click on the **"Body"** tab
   - Select **"raw"** radio button
   - In the dropdown next to "raw", select **"JSON"**
   - Enter your JSON:
   ```json
   {
     "age": 35,
     "income": 50000,
     "loan_amount": 20000
   }
   ```

6. **Send**: Click the blue **"Send"** button

**Expected Response**:
```json
{
  "received_data": {
    "age": 35,
    "income": 50000,
    "loan_amount": 20000
  },
  "prediction": "approved",
  "confidence": 0.85,
  "message": "This is a dummy prediction. ML model coming soon!"
}
```

---

### Test 2: POST /predict (Different Data)

**URL**: `http://localhost:5000/predict`  
**Method**: `POST`  
**Headers**: `Content-Type: application/json`  
**Body (raw JSON)**:
```json
{
  "income": 5000,
  "loan_amount": 150,
  "credit_history": 1
}
```

**Note**: This might fail validation because `age` is missing. That's expected!

---

### Test 3: POST /predict (Different Data - Lower Income)

**URL**: `http://localhost:5000/predict`  
**Method**: `POST`  
**Headers**: `Content-Type: application/json`  
**Body (raw JSON)**:
```json
{
  "income": 3000,
  "loan_amount": 200,
  "credit_history": 0
}
```

---

### Test 4: POST /predict (Empty Body)

**URL**: `http://localhost:5000/predict`  
**Method**: `POST`  
**Headers**: `Content-Type: application/json`  
**Body (raw JSON)**:
```json
{}
```

**Expected**: Should return error "No JSON data provided" or "Missing required fields"

---

## Quick Tips for Postman

1. **Always check Headers tab**: Make sure `Content-Type: application/json` is set
2. **Use raw JSON**: In Body tab, select "raw" and then "JSON" from dropdown
3. **Save requests**: Click "Save" to save your requests for later
4. **Create a Collection**: Organize all your API tests in a collection

---

## 💾 Saving Requests in Postman (IMPORTANT!)

**Answer: No, you don't need to re-enter requests after refreshing!** But you MUST save them first.

### How to Save Requests:

#### Option 1: Save Individual Requests
1. After configuring your request (URL, method, headers, body)
2. Click the **"Save"** button (top right, next to "Send")
3. Choose:
   - **Save to existing collection** (if you have one)
   - **Create new collection** (recommended for first time)
4. Give your request a name (e.g., "GET Home", "POST Predict")
5. Click **"Save"**

#### Option 2: Create a Collection First (Recommended)
1. Click **"Collections"** in the left sidebar
2. Click **"+"** or **"Create Collection"**
3. Name it: **"Loan Predictor API"**
4. Now when you save requests, select this collection

### After Saving:
- ✅ Requests persist after browser refresh
- ✅ Requests persist after closing Postman
- ✅ You can organize requests in folders
- ✅ You can share collections with others

### If You DON'T Save:
- ❌ Requests are lost after refresh
- ❌ You'll need to re-enter everything
- ❌ No way to organize or reuse

### Quick Setup: Create All Requests at Once
1. Create collection: **"Loan Predictor API"**
2. Create folder: **"GET Requests"**
3. Create folder: **"POST Requests"**
4. Save each request in the appropriate folder

**Pro Tip**: Postman also saves your request **History** automatically, but saving to a Collection is better for organization!

---

## Other Endpoints to Test

### GET /api
- **Method**: `GET`
- **URL**: `http://localhost:5000/api`
- **No headers or body needed**
- **Expected**: API info JSON

### GET /health
- **Method**: `GET`
- **URL**: `http://localhost:5000/health`
- **Expected**: `{"status": "healthy"}`

### POST /validate-loan
- **Method**: `POST`
- **URL**: `http://localhost:5000/validate-loan`
- **Headers**: `Content-Type: application/json`
- **Body**:
```json
{
  "age": 25,
  "income": 40000,
  "loan_amount": 15000
}
```

