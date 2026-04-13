# 🐛 TrafficLab Debugging Guide

## Overview
This guide teaches you how to debug the TrafficLab system effectively using various techniques and tools.

## 🔍 Debugging Techniques

### 1. **Console Output Debugging**

The system has extensive print statements with emojis for easy identification:

```bash
# Run the backend to see debug output
cd /home/guy/Projects/Traffic/TrafficLab/backend
python3 main.py
```

**Key Debug Messages:**
- 🔍 Checking database for existing entities...
- 📊 Database status: {...}
- ✅ Loading entities from database...
- 📥 Loading entities from SUMO and saving to database...
- 🚗 Populating vehicles...
- 🐛 DEBUG: System State

### 2. **Built-in Debug Methods**

The system now includes debug methods you can call:

```python
# Debug system state
sumo_simulation.debug_system_state()

# Debug database connection
sumo_simulation.debug_database_connection()
```

### 3. **Database Debugging**

#### Check Database Connection
```bash
# Connect to PostgreSQL directly
psql -h localhost -U postgres -d trafficlab

# List tables
\dt

# Check entity counts
SELECT 
    (SELECT COUNT(*) FROM junctions) as junction_count,
    (SELECT COUNT(*) FROM roads) as road_count,
    (SELECT COUNT(*) FROM zones) as zone_count;
```

#### Check Specific Data
```sql
-- Check junctions
SELECT id, x, y, type, zone FROM junctions LIMIT 5;

-- Check roads
SELECT id, from_junction, to_junction, speed, length, num_lanes FROM roads LIMIT 5;

-- Check zones
SELECT id, description FROM zones;

-- Check zone relationships
SELECT zone_id, edge_id FROM zone_edges LIMIT 10;
SELECT zone_id, junction_id FROM zone_junctions LIMIT 10;
```

### 4. **Python Debugging**

#### Add Breakpoints
```python
import pdb; pdb.set_trace()  # Add this line where you want to debug
```

#### Use Python Debugger
```bash
# Run with debugger
python3 -m pdb main.py
```

#### Add Custom Debug Prints
```python
# Add to any method
print(f"🐛 DEBUG: Variable value = {variable}")
print(f"🐛 DEBUG: List length = {len(my_list)}")
```

### 5. **File System Debugging**

#### Check Configuration Files
```bash
# Check SUMO config
cat /home/guy/Projects/Traffic/TrafficLab/backend/sumo/urban_three_zones.sumocfg

# Check simulation config
cat /home/guy/Projects/Traffic/TrafficLab/backend/sumo/sim.config.json
```

#### Check Log Files
```bash
# Check for any log files
find /home/guy/Projects/Traffic/TrafficLab -name "*.log" -type f

# Check system logs
tail -f /var/log/postgresql/postgresql-*.log
```

### 6. **Network Debugging**

#### Check Database Connection
```bash
# Test PostgreSQL connection
pg_isready -h localhost -p 5432

# Check if port is open
netstat -tlnp | grep 5432
```

#### Check API Endpoints
```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/zones
```

## 🚨 Common Issues and Solutions

### Issue 1: "Database connection failed"
**Symptoms:** Error messages about database connection
**Solutions:**
1. Check if PostgreSQL is running: `sudo systemctl status postgresql`
2. Check database credentials in `.env` file
3. Verify database exists: `psql -l | grep trafficlab`

### Issue 2: "No entities loaded from SUMO"
**Symptoms:** 0 junctions, 0 roads, 0 zones loaded
**Solutions:**
1. Check SUMO config file path
2. Verify SUMO network file exists
3. Check SUMO installation: `sumo --version`

### Issue 3: "No vehicles allocated"
**Symptoms:** Zones loaded but 0 vehicles
**Solutions:**
1. Check simulation config file
2. Verify zone allocation percentages
3. Check if roads have single lanes

### Issue 4: "Database already has data"
**Symptoms:** Unique constraint violations
**Solutions:**
1. Clear database: `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`
2. Or modify code to handle existing data

## 🔧 Debugging Tools

### 1. **Python Debugger (pdb)**
```python
import pdb
pdb.set_trace()  # Set breakpoint
```

### 2. **Logging**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### 3. **Profiling**
```python
import cProfile
cProfile.run('your_function()')
```

### 4. **Memory Debugging**
```python
import tracemalloc
tracemalloc.start()
# Your code here
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
```

## 📊 Performance Debugging

### Monitor System Resources
```bash
# Monitor CPU and memory
htop

# Monitor disk usage
df -h

# Monitor network
iftop
```

### Database Performance
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

## 🎯 Debugging Workflow

1. **Start with console output** - Look for error messages
2. **Check database connection** - Use `debug_database_connection()`
3. **Verify configuration files** - Check paths and content
4. **Add debug prints** - Insert strategic print statements
5. **Use breakpoints** - Step through code execution
6. **Check system resources** - Monitor CPU, memory, disk
7. **Test individual components** - Isolate the problem

## 📝 Debugging Checklist

- [ ] Database is running and accessible
- [ ] Configuration files exist and are valid
- [ ] SUMO is properly installed
- [ ] Python dependencies are installed
- [ ] File permissions are correct
- [ ] Network connectivity is working
- [ ] System resources are sufficient
- [ ] Error messages are clear and actionable

## 🆘 Getting Help

When asking for help, include:
1. **Error messages** (full traceback)
2. **Console output** (relevant sections)
3. **System information** (OS, Python version, etc.)
4. **Steps to reproduce** the issue
5. **What you've already tried**

Remember: Debugging is a skill that improves with practice. Start simple, be systematic, and don't be afraid to add debug prints everywhere!
