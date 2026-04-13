# SmartTransportation Lab - Deployment Guide

**First production deploy (step-by-step):** see **[docs/PRODUCTION-DEPLOY.md](docs/PRODUCTION-DEPLOY.md)**.

## 🚀 Quick Start

### Development (Local)
```bash
# Start development environment
docker-compose up --build

# Or use the deployment script
./deploy.sh development
```

### Production (AWS EC2)
```bash
# Deploy to production
./deploy.sh production
```

## 📋 Prerequisites

### Local Development
- Docker
- Docker Compose
- Git

### AWS EC2 Deployment
- AWS Account
- EC2 Instance (t3.micro recommended)
- Domain name (optional)
- SSL certificate (optional)

## 🛠️ AWS EC2 Setup

### 1. Launch EC2 Instance
- **Instance Type**: t3.micro (free tier eligible)
- **OS**: Ubuntu Server 22.04 LTS
- **Security Group**: Open ports 22, 80, 443, 3000, 8000
- **Key Pair**: Create and download .pem file

### 2. Connect to Instance
```bash
# Set permissions
chmod 400 your-key.pem

# Connect
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 3. Install Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Git
sudo apt install git -y
```

### 4. Deploy Application
```bash
# Clone repository
git clone https://github.com/yourusername/trafficlab.git
cd trafficlab

# Deploy
./deploy.sh production
```

## 🔧 Configuration

### Environment Variables
- **Development**: `.env.development`
- **Production**: `.env.production`

### Ports
- **Frontend**: 3000 (dev) / 80 (prod)
- **Backend**: 8000
- **Database**: 5432
- **Nginx**: 443 (HTTPS)

## 📊 Monitoring

### Check Status
```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart frontend
```

## 🔒 Security

### Firewall (UFW)
```bash
sudo ufw enable
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
```

### SSL Certificate (Let's Encrypt)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d yourdomain.com
```

## 💰 Cost Estimation

### AWS Free Tier (12 months)
- **EC2 t3.micro**: FREE (750 hours/month)
- **Data Transfer**: FREE (1 GB/month)
- **Total**: $0/month

### After Free Tier
- **EC2 t3.micro**: ~$7.50/month
- **Data Transfer**: $0.09/GB
- **Elastic IP**: $3.65/month (optional)
- **Total**: ~$11-15/month

## 🆘 Troubleshooting

### Common Issues
1. **Port already in use**: Stop conflicting services
2. **Permission denied**: Check Docker permissions
3. **Build failed**: Check Dockerfile syntax
4. **Service unhealthy**: Check logs for errors

### Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs frontend
docker-compose logs backend
```

## 📞 Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Restart services: `docker-compose restart`
- Full restart: `docker-compose down && docker-compose up -d`
