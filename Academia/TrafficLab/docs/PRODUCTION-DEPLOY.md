# TrafficLab — first-time production deployment

This guide assumes a **single Linux server** (VPS or cloud VM) with **Docker** and **Docker Compose v2**, and that you deploy from a clone of this repository.

---

## 1. What gets deployed

| Piece | Role |
|--------|------|
| **nginx** | Public entry on **80** (redirect) and **443** (HTTPS). Proxies **`/api/*`**, **`/health`**, **`/docs`** to the backend; everything else to the **frontend** static build. |
| **frontend** | Vue app built to static files, served by nginx **inside** its container (not exposed directly on the host). |
| **backend** | FastAPI + SUMO + PyTorch (CPU). Also reachable on **`127.0.0.1:8000`** on the host for local health checks. |
| **db** | PostgreSQL 15; data in Docker volume **`postgres_data`**. **Not** published to the internet in the default `docker-compose.prod.yml`. |

Compose file: **`docker-compose.prod.yml`**.

---

## 2. Server prerequisites

- **OS:** Ubuntu 22.04/24.04 LTS (or similar).
- **Docker Engine** + **Compose plugin** (`docker compose version` works).
- **Hardware:** for a small deployment, plan at least **4 vCPU / 16 GB RAM /50+ GB SSD** (see project reports for sizing).
- **Firewall:** allow **22** (SSH), **80**, **443**. Do **not** expose PostgreSQL **5432** publicly.

Install Docker (example):

```bash
sudo apt update && sudo apt install -y ca-certificates curl
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
# log out and back in
```

---

## 3. Get the code on the server

```bash
git clone <your-repo-url> TrafficLab
cd TrafficLab
git checkout <branch-you-ship>   # e.g. main or deploy
```

---

## 4. TLS certificates (HTTPS)

Edge nginx expects:

- **`ssl/cert.pem`**
- **`ssl/key.pem`**

in **`./ssl`** at the **project root** (same directory as `docker-compose.prod.yml`).

### Option A — self-signed (testing / internal only)

```bash
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem \
  -subj "/CN=your-server-hostname"
```

Browsers will show a warning until you use a real certificate.

### Option B — Let’s Encrypt (public domain)

Point **A record** for your domain to the server’s public IP. Temporarily serve HTTP validation (or use **certbot** with **standalone** / **DNS**). Typical pattern:

```bash
sudo apt install -y certbot
# obtain certs (method depends on your setup), then copy fullchain + privkey:
# sudo cp /etc/letsencrypt/live/yourdomain/fullchain.pem ssl/cert.pem
# sudo cp /etc/letsencrypt/live/yourdomain/privkey.pem ssl/key.pem
```

Then reload nginx after renewal (automation via cron or certbot hooks).

---

## 5. Secrets and environment

**Do not** use default passwords in real production.

Create a **`.env`** next to `docker-compose.prod.yml` (this file should **not** be committed):

```bash
# Strong DB password
POSTGRES_PASSWORD=your-long-random-password

# Must match POSTGRES_* above
DATABASE_URL=postgresql://user:your-long-random-password@db:5432/trafficlab
```

**CORS:** the backend allows `localhost` / `127.0.0.1` for dev; for your **real site** add:

```bash
CORS_EXTRA_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

Compose passes these into the backend container (see `docker-compose.prod.yml`).

---

## 6. Labels / seed data (optional)

- If **`backend/labels.json`** is missing but **`labels_1.json` … `labels_3.json`** exist, the **backend merges them on first startup** into **`labels.json`** (can take a long time and a lot of RAM).
- **`POST /api/admin/seed-data`** uses **`labels.json`** (or **`labels.json.example`** as fallback).

---

## 7. Build and run

From the repo root:

```bash
docker compose -f docker-compose.prod.yml --env-file .env up -d --build
```

Watch logs until the backend finishes startup (DB init, optional labels merge, SUMO):

```bash
docker compose -f docker-compose.prod.yml logs -f backend
```

---

## 8. Smoke tests

From the **server**:

```bash
# Backend (loopback)
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/health/db

# Through HTTPS (replace host; use -k only for self-signed)
curl -sS -k https://127.0.0.1/health
curl -sS -k https://127.0.0.1/ | head -c 200
```

From your **laptop** (use the server’s DNS or IP):

- Open **`https://your-server/`** (or IP with `-k` / trusted cert).
- Try the demo routes in the SPA; confirm network calls go to **`/api/...`** on the **same host** (production build uses **relative** API URLs unless you set **`VITE_API_BASE_URL`** at build time).

**API docs:** `https://your-server/docs` (proxied to FastAPI).

---

## 9. Operations

```bash
# Status
docker compose -f docker-compose.prod.yml ps

# Logs
docker compose -f docker-compose.prod.yml logs -f

# Restart one service
docker compose -f docker-compose.prod.yml restart backend

# Stop everything
docker compose -f docker-compose.prod.yml down

# Stop and delete database volume (DESTROYS DATA)
docker compose -f docker-compose.prod.yml down -v
```

**Backups:** snapshot or dump PostgreSQL regularly, e.g.:

```bash
docker compose -f docker-compose.prod.yml exec -T db \
  pg_dump -U user trafficlab > backup-$(date -I).sql
```

---

## 10. `deploy.sh` note

The bundled **`deploy.sh`** targets an older workflow (`docker-compose` v1 and dev ports). Prefer the **`docker compose -f docker-compose.prod.yml ...`** commands above unless you update the script for Compose v2 and prod URLs.

---

## 11. Common problems

| Symptom | What to check |
|---------|----------------|
| **502 / empty page** | `docker compose ... logs nginx` and `backend`; confirm **`ssl/`** files exist; confirm **frontend** and **backend** are **Up**. |
| **CORS errors in browser** | Set **`CORS_EXTRA_ORIGINS`** to your exact **`https://`** origin(s). |
| **API 404 behind nginx** | Edge config must forward **`/api/...`** unchanged; see **`nginx/nginx.prod.conf`**. |
| **DB errors / save fails** | `curl http://127.0.0.1:8000/health/db`; verify **`.env`** **`DATABASE_URL`** matches **`POSTGRES_PASSWORD`**. |
| **Very slow first start** | **Labels merge** from **`labels_*.json`**; check backend logs. |

---

## 12. Security checklist (minimal)

- [ ] Strong **`POSTGRES_PASSWORD`** and matching **`DATABASE_URL`**
- [ ] Real TLS for public deployments
- [ ] Firewall: only **22, 80, 443** (no **5432** from internet)
- [ ] Regular **backups** and OS updates
- [ ] Optional: reverse proxy / WAF in front of the VM

---

*After your first successful deploy, keep a short runbook (IP, domain, where `.env` lives, backup command) for your team.*
