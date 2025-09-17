# 🚀 Deployment Guide
## AI-Powered Excel Interviewer System

### Table of Contents
1. [System Requirements](#system-requirements)
2. [Development Environment Setup](#development-environment-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Configuration](#security-configuration)
8. [Performance Optimization](#performance-optimization)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores (8 cores recommended)
- **RAM**: 16GB (32GB recommended)
- **Storage**: 100GB SSD (500GB recommended)
- **Network**: 1Gbps connection
- **OS**: Ubuntu 20.04+ / Windows Server 2019+ / macOS 11+

### Software Dependencies
```bash
# System packages
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev
sudo apt install -y redis-server postgresql-13 nginx
sudo apt install -y docker.io docker-compose
sudo apt install -y supervisor
sudo apt install -y curl wget git

# Python dependencies
pip install -r requirements.txt
```

### Environment Variables
Create `.env` file in project root:
```bash
# Application Configuration
APP_NAME="AI Excel Interviewer"
APP_VERSION="1.0.0"
APP_ENV="production"
APP_DEBUG="false"
APP_SECRET_KEY="your-secret-key-here"

# Database Configuration
DATABASE_URL="postgresql://user:password@localhost:5432/ai_interviewer"
REDIS_URL="redis://localhost:6379/0"

# AI/ML Configuration
OPENAI_API_KEY="your-openai-api-key"
HUGGINGFACE_API_KEY="your-huggingface-api-key"
MODEL_CACHE_DIR="/var/cache/ai_models"

# Security Configuration
JWT_SECRET_KEY="your-jwt-secret-key"
API_RATE_LIMIT="1000"
CORS_ALLOWED_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"

# Monitoring Configuration
SENTRY_DSN="your-sentry-dsn"
LOG_LEVEL="INFO"
METRICS_ENABLED="true"

# External Services
SENDGRID_API_KEY="your-sendgrid-api-key"
AWS_ACCESS_KEY_ID="your-aws-access-key"
AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
AWS_REGION="us-east-1"
```

---

## Development Environment Setup

### Local Development Setup
```bash
# Clone repository
git clone https://github.com/your-org/ai-excel-interviewer.git
cd ai-excel-interviewer

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup database
python manage.py migrate
python manage.py createsuperuser

# Load initial data
python manage.py loaddata initial_data.json

# Run development server
python manage.py runserver
```

### Development Tools Setup
```bash
# Install pre-commit hooks
pre-commit install

# Setup IDE configuration
# Copy .vscode/settings.json.template to .vscode/settings.json
# Copy .idea/runConfigurations template files

# Setup environment variables
cp .env.example .env
# Edit .env with your local settings
```

### Testing Environment
```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/

# Run security tests
pytest tests/security/

# Generate coverage report
pytest --cov=src --cov-report=html
```

---

## Production Deployment

### Server Setup
```bash
# Create application user
sudo useradd -m -s /bin/bash ai_interviewer
sudo usermod -aG sudo ai_interviewer

# Setup directory structure
sudo mkdir -p /opt/ai_interviewer
sudo chown ai_interviewer:ai_interviewer /opt/ai_interviewer

# Setup virtual environment
sudo -u ai_interviewer python3.9 -m venv /opt/ai_interviewer/venv

# Clone and setup application
cd /opt/ai_interviewer
sudo -u ai_interviewer git clone https://github.com/your-org/ai-excel-interviewer.git .
sudo -u ai_interviewer /opt/ai_interviewer/venv/bin/pip install -r requirements.txt
```

### Database Setup
```bash
# Create PostgreSQL database
sudo -u postgres createdb ai_interviewer
sudo -u postgres psql -c "CREATE USER ai_interviewer WITH PASSWORD 'secure_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ai_interviewer TO ai_interviewer;"

# Run migrations
sudo -u ai_interviewer /opt/ai_interviewer/venv/bin/python manage.py migrate

# Create superuser
sudo -u ai_interviewer /opt/ai_interviewer/venv/bin/python manage.py createsuperuser
```

### Redis Setup
```bash
# Configure Redis
sudo cp config/redis.conf /etc/redis/redis.conf
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli ping
```

### Gunicorn Configuration
Create `/etc/systemd/system/ai_interviewer.service`:
```ini
[Unit]
Description=AI Excel Interviewer
After=network.target

[Service]
User=ai_interviewer
Group=ai_interviewer
WorkingDirectory=/opt/ai_interviewer
Environment=PATH=/opt/ai_interviewer/venv/bin
ExecStart=/opt/ai_interviewer/venv/bin/gunicorn \
    --bind unix:/run/ai_interviewer/gunicorn.sock \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --log-level info \
    --access-logfile /var/log/ai_interviewer/access.log \
    --error-logfile /var/log/ai_interviewer/error.log \
    main:app

[Install]
WantedBy=multi-user.target
```

### Nginx Configuration
Create `/etc/nginx/sites-available/ai_interviewer`:
```nginx
upstream ai_interviewer {
    server unix:/run/ai_interviewer/gunicorn.sock fail_timeout=0;
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/ssl/certs/yourdomain.com.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    client_max_body_size 100M;
    
    location / {
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_pass http://ai_interviewer;
    }

    location /static/ {
        alias /opt/ai_interviewer/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location /media/ {
        alias /opt/ai_interviewer/media/;
        expires 7d;
        add_header Cache-Control "public";
    }
}
```

### SSL Certificate Setup
```bash
# Using Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## Docker Deployment

### Docker Compose Configuration
Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://ai_interviewer:password@db:5432/ai_interviewer
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./static:/app/static
      - ./media:/app/media
      - ./logs:/app/logs

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=ai_interviewer
      - POSTGRES_USER=ai_interviewer
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - app

  celery:
    build: .
    command: celery -A main worker -l info
    environment:
      - DATABASE_URL=postgresql://ai_interviewer:password@db:5432/ai_interviewer
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs

  celery-beat:
    build: .
    command: celery -A main beat -l info
    environment:
      - DATABASE_URL=postgresql://ai_interviewer:password@db:5432/ai_interviewer
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs

volumes:
  postgres_data:
  redis_data:
```

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "main:app"]
```

### Docker Deployment Commands
```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f app

# Scale application
docker-compose up -d --scale app=3

# Update application
docker-compose pull
docker-compose up -d

# Backup database
docker-compose exec db pg_dump -U ai_interviewer ai_interviewer > backup.sql
```

---

## Cloud Deployment

### AWS Deployment

#### AWS ECS Configuration
```json
{
  "family": "ai-interviewer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-registry/ai-interviewer:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "APP_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ai-interviewer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### AWS RDS Setup
```bash
# Create RDS instance
aws rds create-db-instance \
    --db-instance-identifier ai-interviewer-db \
    --db-instance-class db.t3.medium \
    --engine postgres \
    --master-username ai_interviewer \
    --master-user-password secure_password \
    --allocated-storage 20 \
    --backup-retention-period 7 \
    --multi-az

# Create read replica
aws rds create-db-instance-read-replica \
    --db-instance-identifier ai-interviewer-db-replica \
    --source-db-instance-identifier ai-interviewer-db
```

#### AWS ElastiCache Setup
```bash
# Create Redis cluster
aws elasticache create-cache-cluster \
    --cache-cluster-id ai-interviewer-redis \
    --cache-node-type cache.t3.micro \
    --engine redis \
    --num-cache-nodes 1 \
    --security-group-ids sg-xxxxxxxxx
```

### Google Cloud Platform Deployment

#### GKE Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-interviewer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-interviewer
  template:
    metadata:
      labels:
        app: ai-interviewer
    spec:
      containers:
      - name: app
        image: gcr.io/your-project/ai-interviewer:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-interviewer-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ai-interviewer-service
spec:
  selector:
    app: ai-interviewer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Azure Deployment

#### Azure Container Instances
```bash
# Create resource group
az group create --name ai-interviewer-rg --location eastus

# Create container instance
az container create \
    --resource-group ai-interviewer-rg \
    --name ai-interviewer-app \
    --image your-registry/ai-interviewer:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --environment-variables \
        APP_ENV=production \
        DATABASE_URL=your-database-url \
    --registry-login-server your-registry.azurecr.io \
    --registry-username your-username \
    --registry-password your-password
```

---

## Monitoring and Logging

### Application Monitoring
```python
# Install monitoring dependencies
pip install prometheus-client sentry-sdk

# Configure Sentry
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.1,
    send_default_pii=True
)

# Configure Prometheus
from prometheus_client import Counter, Histogram, start_http_server

request_count = Counter('app_requests_total', 'Total requests')
request_latency = Histogram('app_request_latency_seconds', 'Request latency')

# Start metrics server
start_http_server(8001)
```

### Log Configuration
Create `config/logging.py`:
```python
import logging
import logging.config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai_interviewer/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai_interviewer/error.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'json',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file', 'error_file'],
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'ai_interviewer': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False,
        },
    },
}
```

### Health Checks
Create `health_check.py`:
```python
import asyncio
import aioredis
import asyncpg
from typing import Dict, Any

async def check_database() -> Dict[str, Any]:
    try:
        conn = await asyncpg.connect(dsn=os.getenv('DATABASE_URL'))
        await conn.execute('SELECT 1')
        await conn.close()
        return {"status": "healthy", "latency_ms": 5}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_redis() -> Dict[str, Any]:
    try:
        redis = await aioredis.create_redis_pool(os.getenv('REDIS_URL'))
        await redis.ping()
        redis.close()
        await redis.wait_closed()
        return {"status": "healthy", "latency_ms": 2}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

async def check_external_services() -> Dict[str, Any]:
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "openai": await check_openai(),
        "disk_space": check_disk_space(),
        "memory": check_memory_usage()
    }
    
    overall_status = "healthy" if all(
        check["status"] == "healthy" for check in checks.values()
    ) else "unhealthy"
    
    return {
        "status": overall_status,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }
```

---

## Security Configuration

### Security Headers
Configure security headers in Nginx:
```nginx
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com;" always;
```

### Database Security
```bash
# Create secure PostgreSQL configuration
echo "local   all             all                                     md5" >> /etc/postgresql/13/main/pg_hba.conf
echo "host    all             all             127.0.0.1/32            md5" >> /etc/postgresql/13/main/pg_hba.conf
echo "host    all             all             ::1/128                 md5" >> /etc/postgresql/13/main/pg_hba.conf

# Enable SSL
sudo -u postgres openssl req -new -x509 -days 365 -nodes -text -out /var/lib/postgresql/server.crt -keyout /var/lib/postgresql/server.key
sudo -u postgres chmod 600 /var/lib/postgresql/server.key
sudo -u postgres chown postgres:postgres /var/lib/postgresql/server.key

# Configure PostgreSQL for SSL
echo "ssl = on" >> /etc/postgresql/13/main/postgresql.conf
echo "ssl_cert_file = '/var/lib/postgresql/server.crt'" >> /etc/postgresql/13/main/postgresql.conf
echo "ssl_key_file = '/var/lib/postgresql/server.key'" >> /etc/postgresql/13/main/postgresql.conf
```

### API Security
```python
# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour", "100 per minute"]
)

# JWT configuration
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)

# Input validation
from marshmallow import Schema, fields, validate

class InterviewStartSchema(Schema):
    candidate_id = fields.Str(required=True, validate=validate.Length(min=1, max=100))
    position_level = fields.Str(required=True, validate=validate.OneOf(['junior', 'mid', 'senior', 'expert']))
    focus_areas = fields.List(fields.Str(), validate=validate.Length(min=1, max=10))
```

---

## Performance Optimization

### Application Optimization
```python
# Database connection pooling
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT'),
        'CONN_MAX_AGE': 60,
        'OPTIONS': {
            'MAX_CONNS': 20,
            'MIN_CONNS': 5,
        }
    }
}

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {'max_connections': 50}
        }
    }
}

# Static files optimization
STATICFILES_STORAGE = 'django.contrib.staticfiles.storage.ManifestStaticFilesStorage'
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]
```

### Database Optimization
```sql
-- Create indexes for common queries
CREATE INDEX idx_interviews_candidate_id ON interviews(candidate_id);
CREATE INDEX idx_interviews_created_at ON interviews(created_at);
CREATE INDEX idx_evaluations_session_id ON evaluations(session_id);
CREATE INDEX idx_evaluations_score ON evaluations(overall_score);

-- Optimize table structure
ALTER TABLE interviews SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE evaluations SET (autovacuum_vacuum_scale_factor = 0.1);

-- Create materialized views for analytics
CREATE MATERIALIZED VIEW dashboard_metrics AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as total_interviews,
    AVG(overall_score) as average_score,
    COUNT(CASE WHEN status = 'completed' THEN 1 END)::float / COUNT(*) as completion_rate
FROM interviews
GROUP BY DATE(created_at);

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_dashboard_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY dashboard_metrics;
END;
$$ LANGUAGE plpgsql;
```

### CDN Configuration
```nginx
# Cloudflare CDN configuration
location ~* \.(jpg|jpeg|png|gif|ico|css|js|woff|woff2|ttf|svg|eot)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header Vary "Accept-Encoding";
    
    # Cloudflare headers
    add_header CF-Cache-Status "HIT" always;
    add_header CF-Ray "$http_cf_ray" always;
}

# API response caching
location /api/ {
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;
    proxy_cache_valid 404 1m;
    proxy_cache_key "$scheme$request_method$host$request_uri";
    proxy_cache_use_stale error timeout updating http_500 http_502 http_503 http_504;
    
    proxy_pass http://ai_interviewer;
}
```

---

## Backup and Recovery

### Database Backup
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/postgresql"
DB_NAME="ai_interviewer"
DB_USER="ai_interviewer"

# Create backup directory
mkdir -p $BACKUP_DIR

# Full database backup
pg_dump -U $DB_USER -h localhost $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Backup retention (keep last 30 days)
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE.sql.gz s3://your-backup-bucket/postgresql/
```

### Application Backup
```bash
# Application files backup
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/application"
APP_DIR="/opt/ai_interviewer"

# Create backup
tar -czf $BACKUP_DIR/app_backup_$DATE.tar.gz -C $APP_DIR .

# Backup media files separately
tar -czf $BACKUP_DIR/media_backup_$DATE.tar.gz -C $APP_DIR/media .

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/application/
```

### Disaster Recovery Plan
```bash
# Recovery script
#!/bin/bash
# Restore from backup

# Stop application
sudo systemctl stop ai_interviewer

# Restore database
gunzip < /backup/postgresql/backup_latest.sql.gz | psql -U ai_interviewer -d ai_interviewer

# Restore application files
cd /opt/ai_interviewer
sudo tar -xzf /backup/application/app_backup_latest.tar.gz

# Restore media files
sudo tar -xzf /backup/application/media_backup_latest.tar.gz -C media/

# Restart application
sudo systemctl start ai_interviewer

# Verify health
curl -f http://localhost/health || exit 1
```

---

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
sudo journalctl -u ai_interviewer -f

# Check file permissions
sudo chown -R ai_interviewer:ai_interviewer /opt/ai_interviewer
sudo chmod -R 755 /opt/ai_interviewer

# Check Python dependencies
/opt/ai_interviewer/venv/bin/pip check

# Test configuration
/opt/ai_interviewer/venv/bin/python manage.py check
```

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -U ai_interviewer -d ai_interviewer -c "SELECT 1;"

# Check connection pool
sudo -u ai_interviewer /opt/ai_interviewer/venv/bin/python -c "
import django
django.setup()
from django.db import connection
print('Connection test:', connection.ensure_connection())
"
```

#### Memory Issues
```bash
# Monitor memory usage
htop
free -h

# Check for memory leaks
sudo -u ai_interviewer /opt/ai_interviewer/venv/bin/python -m memory_profiler your_script.py

# Optimize Gunicorn workers
# Reduce workers if memory is constrained
# workers = multiprocessing.cpu_count() * 2 + 1
```

#### Performance Issues
```bash
# Check slow queries
sudo -u postgres psql -d ai_interviewer -c "
SELECT query, mean_time, calls, total_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"

# Check Redis performance
redis-cli --latency
redis-cli info stats

# Monitor application performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost/api/health/
```

### Log Analysis
```bash
# Analyze error logs
sudo tail -f /var/log/ai_interviewer/error.log | grep ERROR

# Count error types
sudo grep -c "ERROR\|WARNING\|CRITICAL" /var/log/ai_interviewer/error.log

# Find specific errors
sudo grep -i "database\|connection\|timeout" /var/log/ai_interviewer/error.log

# Monitor real-time logs
sudo multitail -s 2 /var/log/ai_interviewer/access.log /var/log/ai_interviewer/error.log
```

### Performance Monitoring
```bash
# System performance
iostat -x 1
vmstat 1
sar -u 1

# Network performance
netstat -tuln
ss -tuln
iftop -i eth0

# Application performance
ab -n 1000 -c 10 http://localhost/api/health/
wrk -t12 -c400 -d30s http://localhost/api/health/
```

---

## Support and Maintenance

### Regular Maintenance Tasks
- **Daily**: Monitor logs, check system health, verify backups
- **Weekly**: Update dependencies, review performance metrics, clean old logs
- **Monthly**: Security updates, database optimization, capacity planning
- **Quarterly**: Disaster recovery testing, security audit, performance review

### Emergency Contacts
- **Technical Support**: tech-support@yourcompany.com
- **System Administration**: sysadmin@yourcompany.com
- **Security Team**: security@yourcompany.com
- **On-call Engineer**: +1-xxx-xxx-xxxx

### Documentation Links
- [Technical Architecture](TECHNICAL_ARCHITECTURE.md)
- [API Documentation](API_DOCUMENTATION.md)
- [System Monitoring Dashboard](https://monitoring.yourdomain.com)
- [Status Page](https://status.yourdomain.com)

---

**Last Updated**: January 2024  
**Version**: 1.0.0  
**Maintained By**: AI Interviewer Team