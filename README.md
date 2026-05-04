# Image Search Service

Vector-based image similarity search using CLIP + Qdrant.

---

## Requirements

### Local Development
- Python 3.10+
- Qdrant running on `localhost:6333`
- 4GB+ RAM (CLIP model needs ~1.5GB)

### Production Server
- Ubuntu 22.04 VPS (DigitalOcean, AWS EC2, Hetzner, etc.)
- 4GB+ RAM, 2 CPU cores
- Docker installed

---

## Project Structure

```
imageSearch/
├── main.py              # FastAPI app + all endpoints
├── clip_encoder.py      # CLIP model loader + image encoder
├── qdrant_service.py    # Qdrant client + vector operations
├── requirements.txt     # Python dependencies
├── qdrant_storage/      # Qdrant database (auto-created)
└── README.md
```

---

## Python Dependencies

Install inside a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:
```
fastapi
uvicorn[standard]
qdrant-client
git+https://github.com/openai/CLIP.git
torch
torchvision
Pillow
python-multipart
ftfy
regex
```

---

## Run Locally

**Step 1 — Start Qdrant**
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

**Step 2 — Start API server**
```bash
source venv/bin/activate
python -m uvicorn main:app --reload
```

Server runs at: `http://127.0.0.1:8000`

---

## API Endpoints

### `POST /upload`
Upload a product image.

| Field | Type | Description |
|-------|------|-------------|
| `product_id` | string (form) | Unique product identifier |
| `status` | int (form) | `1` = active, `0` = inactive |
| `image` | file (form) | JPG / PNG / WEBP image |

```bash
curl -X POST http://localhost:8000/upload \
  -F "product_id=prod_001" \
  -F "status=1" \
  -F "image=@photo.jpg"
```

---

### `PATCH /status/{product_id}`
Activate or deactivate a product.

```bash
curl -X PATCH http://localhost:8000/status/prod_001 \
  -H "Content-Type: application/json" \
  -d '{"status": 0}'
```

---

### `POST /search`
Search for similar active products.

| Field | Type | Description |
|-------|------|-------------|
| `image` | file (form) | Query image |
| `threshold` | float (query, optional) | Minimum similarity score. Default: `0.7` |

```bash
curl -X POST "http://localhost:8000/search?threshold=0.7" \
  -F "image=@query.jpg"
```

Response:
```json
{
  "results": [
    {"product_id": "prod_001", "score": 1.0},
    {"product_id": "prod_003", "score": 0.795}
  ]
}
```

> Only products with `status=1` appear in results.

---

## Score Guide

| Score | Meaning |
|-------|---------|
| 0.9 – 1.0 | Near-identical |
| 0.75 – 0.9 | Same category / visually similar |
| 0.6 – 0.75 | Loosely related |
| < 0.6 | Unrelated |

---

## Deploy to Production (Ubuntu 22.04)

### 1. Install system packages

```bash
apt update && apt install -y python3 python3-pip python3-venv git docker.io nginx
systemctl enable docker && systemctl start docker
```

### 2. Upload project

```bash
# From your local machine
scp -r /Users/rarashed/Desktop/imageSearch root@your-server-ip:/app/imageSearch
```

Or via Git:
```bash
# Local
git init && git add . && git commit -m "init"
git remote add origin https://github.com/yourname/image-search.git
git push

# Server
git clone https://github.com/yourname/image-search.git /app/imageSearch
```

### 3. Start Qdrant with Docker

```bash
docker run -d \
  --name qdrant \
  --restart always \
  -p 6333:6333 \
  -v /app/imageSearch/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 4. Setup Python environment

```bash
cd /app/imageSearch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5. Create systemd service

```bash
nano /etc/systemd/system/imagesearch.service
```

```ini
[Unit]
Description=Image Search API
After=network.target

[Service]
User=root
WorkingDirectory=/app/imageSearch
ExecStart=/app/imageSearch/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable imagesearch
systemctl start imagesearch
```

### 6. Nginx reverse proxy (optional)

```bash
nano /etc/nginx/sites-available/imagesearch
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/imagesearch /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx
```

---

## Database Management

| Task | Command |
|------|---------|
| Stop Qdrant | `docker stop qdrant` |
| Start Qdrant | `docker start qdrant` |
| View Qdrant logs | `docker logs qdrant -f` |
| Backup database | `cp -r /app/imageSearch/qdrant_storage /backup/qdrant_$(date +%Y%m%d)` |
| Restore backup | `cp -r /backup/qdrant_20240501 /app/imageSearch/qdrant_storage` |
| Qdrant dashboard | `http://your-server-ip:6333/dashboard` |

---

## Logs & Monitoring

```bash
# API server logs
journalctl -u imagesearch -f

# Check service status
systemctl status imagesearch

# Restart service
systemctl restart imagesearch
```

---

## URLs

| URL | Description |
|-----|-------------|
| `http://your-domain.com/` | Landing page + system status |
| `http://your-domain.com/docs` | Swagger UI (interactive API docs) |
| `http://your-domain.com/redoc` | ReDoc API reference |
| `http://your-domain.com:6333/dashboard` | Qdrant database dashboard |
