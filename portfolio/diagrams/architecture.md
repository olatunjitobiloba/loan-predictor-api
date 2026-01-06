# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                        │
├─────────────────────────────────────────────────────────────|
│  Web Browser  │  Mobile App  │  API Clients  │  Postman     │
└────────┬────────────────┬────────────┬────────────┬─────────┘
         │                │            │            │
         └────────────────┴────────────┴────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  Frontend (HTML/CSS/JS)  │  Swagger UI  │  REST Endpoints   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                      │
├─────────────────────────────────────────────────────────────┤
│                      Flask Application                      │
│  ┌──────────────┬──────────────┬──────────────────────┐     │
│  │  Routing     │  Validation  │  Rate Limiting       │     │
│  │  Caching     │  Logging     │  Error Handling      │     │
│  └──────────────┴──────────────┴──────────────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────┐
│   ML LAYER  │  │  DATA LAYER  │  │  CACHE LAYER │
├─────────────┤  ├──────────────┤  ├──────────────┤
│ 4 ML Models │  │  PostgreSQL  │  │  Flask-Cache │
│ - Random    │  │  Database    │  │  (In-Memory) │
│   Forest    │  │              │  │              │
│ - Logistic  │  │  Tables:     │  │  Cached:     │
│   Regression│  │  - Prediction│  │  - Stats     │
│ - Gradient  │  │  - DailyStats│  │  - Analytics │
│   Boosting  │  │              │  │  - Model Info│
│ - SVM       │  │              │  │              │
└─────────────┘  └──────────────┘  └──────────────┘
         │               │               │
         └───────────────┴───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  Render (Platform)  │  Gunicorn (WSGI)  │  PostgreSQL (DB)  │
└─────────────────────────────────────────────────────────────┘
```

## Request Flow

```
User Request
     ↓
Flask Routing
     ↓
Rate Limiting Check
     ↓
Cache Check (if GET)
     ↓
Input Validation
     ↓
Data Preprocessing
     ↓
ML Model Prediction
     ↓
Database Storage
     ↓
Response Formatting
     ↓
Compression & Return
```

## Data Flow

```
┌──────────────┐
│  User Input  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│   Validation     │
│   - Type checks  │
│   - Range checks │
│   - Format checks│
└──────┬───────────┘
       │
       ▼
┌──────────────────────────┐
│   Feature Engineering    │
│   - Total Income         │
│   - Loan/Income Ratio    │
│   - Log Transformations  │
│   - One-Hot Encoding     │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────┐
│   ML Models      │
│   (4 models)     │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Prediction     │
│   + Confidence   │
│   + Probability  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   Database       │
│   Storage        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│   JSON Response  │
└──────────────────┘
```

## Technology Stack

### Backend
- **Framework:** Flask 3.0
- **WSGI Server:** Gunicorn
- **ML Library:** scikit-learn 1.3
- **Database ORM:** SQLAlchemy
- **Validation:** Custom validators
- **Caching:** Flask-Caching
- **Rate Limiting:** Flask-Limiter
- **Compression:** Flask-Compress
- **Documentation:** Flasgger (Swagger/OpenAPI)

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Custom styling, Grid, Flexbox
- **JavaScript (ES6+)** - Vanilla JS, Fetch API
- **Responsive Design** - Mobile-first approach

### Database
- **Production:** PostgreSQL 14
- **Development:** SQLite
- **Tables:** Predictions, DailyStats

### ML Models
- **Random Forest:** 100 estimators, max_depth=10
- **Logistic Regression:** L2 regularization
- **Gradient Boosting:** 100 estimators, learning_rate=0.1
- **SVM:** RBF kernel, probability=True

### Infrastructure
- **Hosting:** Render
- **Database:** Render PostgreSQL
- **CI/CD:** Git push auto-deploy
- **Monitoring:** Built-in performance tracking

## Performance Optimizations

### Caching Strategy
| Endpoint          | Cache Time | Benefit      |
|-------------------|------------|--------------|
| /statistics       | 60s        | 85% hit rate |
| /analytics        | 120s       | 80% hit rate |
| /model-info       | 3600s      | 95% hit rate |
| /history          | 30s        | 70% hit rate |

### Rate Limiting
| Endpoint          | Limit                  |
|-------------------|------------------------|
| Default           | 200/hour, 1000/day     |
| /predict          | 100/hour               |
| /models/benchmark | 10/hour                |

### Response Compression
- Gzip compression enabled
- ~70% size reduction
- Automatic for responses > 500 bytes

## Security Features

- Input validation on all endpoints
- SQL injection prevention (SQLAlchemy ORM)
- Rate limiting per IP
- HTTPS enforced in production
- Error message sanitization
- Request logging and monitoring

## Scalability Considerations

### Current Capacity
- ~35 requests/second
- ~150ms average response time
- Single instance deployment

### Future Scaling Options
1. **Horizontal Scaling**
   - Multiple instances behind load balancer
   - Shared PostgreSQL database
   - Redis for distributed caching

2. **Database Optimization**
   - Read replicas for analytics
   - Connection pooling
   - Query optimization with indexes

3. **Caching Enhancement**
   - Redis cache cluster
   - CDN for static assets
   - Edge caching

4. **Async Processing**
   - Celery for background tasks
   - Message queue (RabbitMQ/Redis)
   - Async model predictions

## Monitoring & Observability

### Current Metrics
- Request count
- Response times (avg, p95, p99)
- Error rates
- Cache hit rates
- Model prediction times

### Logging
- Application logs (api.log)
- Request/response logging
- Error tracking
- Performance tracking

## Deployment Pipeline

```
Code Push (GitHub)
     ↓
Render Detects Change
     ↓
Build Environment
     ↓
Install Dependencies
     ↓
Run Migrations (if any)
     ↓
Start Gunicorn
     ↓
Health Check
     ↓
Route Traffic
     ↓
Monitor
```

## API Versioning

- Current: v7.0
- Version in response headers
- Backward compatibility maintained
- Deprecation notices for breaking changes