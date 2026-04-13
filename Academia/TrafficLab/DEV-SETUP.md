# Development Environment Setup

This document explains how to quickly start the TrafficLab development environment after a system restart.

## Quick Start

### Option 1: Docker Compose (Recommended)

The easiest way to start the development environment:

```bash
# Linux/macOS
./dev-start.sh docker

# Windows
dev-start.bat docker
```

This will:
- Start PostgreSQL database
- Build and start the FastAPI backend
- Build and start the Vue.js frontend
- Make all services available on their respective ports

### Option 2: Local Development

If you prefer to run services locally without Docker:

```bash
# Linux/macOS
./dev-start.sh local

# Windows
dev-start.bat local
```

**Prerequisites for local development:**
- Python 3.8+
- Node.js 16+
- PostgreSQL running on localhost:5432

## Services

Once started, the following services will be available:

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Vue.js development server |
| Backend API | http://localhost:8000 | FastAPI backend with auto-reload |
| Database | localhost:5432 | PostgreSQL database |

## Database Credentials

- **Host**: localhost
- **Port**: 5432
- **Database**: trafficlab
- **Username**: user
- **Password**: password

## Stopping Services

### Docker Compose
```bash
docker-compose down
```

### Local Development
- Press `Ctrl+C` in the terminal where you started the script
- Or kill the processes manually

## Troubleshooting

### Port Already in Use
If you get port conflicts:
```bash
# Check what's using the port
lsof -i :3000  # Frontend
lsof -i :8000  # Backend
lsof -i :5432  # Database

# Kill the process
kill -9 <PID>
```

### Docker Issues
```bash
# Clean up Docker containers and images
docker-compose down
docker system prune -a

# Rebuild everything
docker-compose up --build
```

### Database Issues
```bash
# Reset the database
docker-compose down -v  # This removes volumes
docker-compose up --build
```

## Manual Setup (if scripts fail)

### Backend
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
python init_db.py
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Database
```bash
# Using Docker
docker run -d --name trafficlab-db \
  -e POSTGRES_DB=trafficlab \
  -e POSTGRES_USER=user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  postgres:15-alpine

# Or install PostgreSQL locally
# Ubuntu/Debian: sudo apt install postgresql postgresql-contrib
# macOS: brew install postgresql
# Windows: Download from https://www.postgresql.org/download/
```

## Development Tips

1. **Hot Reload**: Both frontend and backend support hot reload during development
2. **API Documentation**: Visit http://localhost:8000/docs for interactive API documentation
3. **Database Management**: Use pgAdmin or any PostgreSQL client to manage the database
4. **Logs**: Check Docker logs with `docker-compose logs -f` or terminal output for local development

## Need Help?

- Check the main [README.md](README.md) for detailed project information
- Review the [DEBUG_GUIDE.md](DEBUG_GUIDE.md) for troubleshooting
- Check the [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
