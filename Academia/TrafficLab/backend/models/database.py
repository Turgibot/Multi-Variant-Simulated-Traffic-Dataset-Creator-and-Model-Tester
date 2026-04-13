from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os
import json

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/trafficlab")

# Create database engine (pre_ping + rollback on return avoid stale aborted transactions from the pool)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_reset_on_return="rollback",
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def _debug_log_database_connect_settings(db_url: str) -> None:
    """Log DB connection details. Set TRAFFICLAB_DEBUG_DB=1 (redacted) or full (includes repr(password))."""
    from sqlalchemy.engine.url import make_url

    mode = os.getenv("TRAFFICLAB_DEBUG_DB", "").strip().lower()
    if mode not in ("1", "true", "yes", "full"):
        return
    try:
        u = make_url(db_url)
        safe = u.render_as_string(hide_password=True)
        pw = u.password
        print(
            f"DEBUG TRAFFICLAB_DEBUG_DB: url={safe} user={u.username!r} host={u.host!r} "
            f"port={u.port!r} database={u.database!r} password_is_none={pw is None} "
            f"password_len={len(pw or '')}"
        )
        if mode == "full":
            print(f"DEBUG TRAFFICLAB_DEBUG_DB full: password repr={pw!r}")
    except Exception as e:
        print(f"DEBUG TRAFFICLAB_DEBUG_DB: could not parse DATABASE_URL: {e!r}")


class Journey(Base):
    """
    Database model for storing journey data
    """
    __tablename__ = "journeys"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    journey_number = Column(Integer, nullable=False, unique=True, index=True)
    vehicle_id = Column(String, nullable=False, index=True)
    start_edge = Column(String, nullable=False)
    end_edge = Column(String, nullable=False)
    route_edges = Column(Text)  # JSON string of route edges
    start_time = Column(Integer, nullable=False)  # Simulation step when journey started
    start_time_string = Column(String, nullable=False)  # Human readable start time
    end_time = Column(Integer, nullable=True)  # Simulation step when journey ended
    end_time_string = Column(String, nullable=True)  # Human readable end time
    distance = Column(Float, nullable=False)  # Total distance in meters
    predicted_eta = Column(Integer, nullable=False)  # Predicted ETA in seconds
    actual_duration = Column(Integer, nullable=True)  # Actual duration in seconds
    absolute_error = Column(Integer, nullable=True)  # Absolute error in seconds
    accuracy = Column(Float, nullable=True)  # Accuracy percentage
    status = Column(String, nullable=False, default="running")  # running, finished, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            "id": self.id,
            "journey_number": self.journey_number,
            "vehicle_id": self.vehicle_id,
            "start_edge": self.start_edge,
            "end_edge": self.end_edge,
            "route_edges": json.loads(self.route_edges) if self.route_edges else [],
            "start_time": self.start_time,
            "start_time_string": self.start_time_string,
            "end_time": self.end_time,
            "end_time_string": self.end_time_string,
            "distance": self.distance,
            "predicted_eta": self.predicted_eta,
            "actual_duration": self.actual_duration,
            "absolute_error": self.absolute_error,
            "accuracy": self.accuracy,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

def create_database():
    """Create the database if it doesn't exist"""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    from sqlalchemy.engine.url import make_url

    try:
        db_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/trafficlab")
        url = make_url(db_url)
        if not url.host:
            raise RuntimeError(
                "DATABASE_URL has no host (expected e.g. postgresql://user:PASSWORD@db:5432/trafficlab). "
                "Common causes: a '#' in the password/URL without quoting the line in .env, or a truncated URL "
                "like postgresql://user: — check /opt/TrafficLab/.env on the server."
            )
        if url.host == "user":
            raise RuntimeError(
                "DATABASE_URL is parsed with host 'user' (wrong). You likely have postgresql://user: with a "
                "missing password or everything after '#' was dropped as a comment in .env. Use a full URL: "
                "postgresql://user:YOUR_PASSWORD@db:5432/trafficlab"
            )

        _debug_log_database_connect_settings(db_url)

        conn = psycopg2.connect(
            host=url.host,
            port=url.port or 5432,
            user=url.username,
            password=url.password,
            dbname="postgres",
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()

        db_name = url.database or "trafficlab"
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}';")
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(f'CREATE DATABASE {db_name};')
            print(f'✅ Database {db_name} created successfully')
        else:
            print(f'✅ Database {db_name} already exists')
        
        # Close connection
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f'❌ Error creating database: {e}')
        raise

def create_tables():
    """Create all database tables"""
    # First ensure the database exists
    create_database()
    
    # Then create the tables
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session; rollback on errors so the connection is not left in aborted state."""
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# Journey CRUD operations

def get_next_journey_number(db: Session):
    """Get the next journey number for continuous numbering."""
    last_journey = db.query(Journey).order_by(Journey.journey_number.desc()).first()
    if last_journey:
        return last_journey.journey_number + 1
    return 1

def create_journey(db: Session, journey_data: dict):
    """Create a new journey in the database"""
    try:
        allowed = {
            'vehicle_id', 'start_edge', 'end_edge', 'route_edges',
            'start_time', 'start_time_string', 'end_time', 'end_time_string',
            'distance', 'predicted_eta', 'actual_duration', 'absolute_error',
            'accuracy', 'status',
        }
        data = {k: v for k, v in journey_data.items() if k in allowed}

        def _as_int(v):
            if v is None:
                return None
            return int(round(float(v)))

        for key in ('start_time', 'end_time', 'predicted_eta', 'actual_duration', 'absolute_error'):
            if key in data and data[key] is not None:
                data[key] = _as_int(data[key])

        if 'distance' in data and data['distance'] is not None:
            data['distance'] = float(data['distance'])
        if 'accuracy' in data and data['accuracy'] is not None:
            data['accuracy'] = float(data['accuracy'])

        if data.get('predicted_eta') is None:
            data['predicted_eta'] = 0

        # Get the next journey number
        journey_number = get_next_journey_number(db)
        data['journey_number'] = journey_number

        # Convert route_edges to JSON string if it's a list
        if 'route_edges' in data and isinstance(data['route_edges'], list):
            data['route_edges'] = json.dumps(data['route_edges'])

        journey = Journey(**data)
        db.add(journey)
        db.commit()
        db.refresh(journey)
        return journey
    except Exception as e:
        print(f"Error creating journey: {e}")
        db.rollback()
        raise

def get_journey(db: Session, journey_id: int):
    """Get a journey by ID"""
    try:
        return db.query(Journey).filter(Journey.id == journey_id).first()
    except Exception as e:
        print(f"Error getting journey: {e}")
        return None

def get_journey_by_number(db: Session, journey_number: int):
    """Get a journey by journey number"""
    try:
        return db.query(Journey).filter(Journey.journey_number == journey_number).first()
    except Exception as e:
        print(f"Error getting journey by number: {e}")
        return None

def get_journey_by_vehicle_id(db: Session, vehicle_id: str):
    """Get a journey by vehicle ID"""
    try:
        return db.query(Journey).filter(Journey.vehicle_id == vehicle_id).first()
    except Exception as e:
        print(f"Error getting journey by vehicle ID: {e}")
        return None

def update_journey(db: Session, journey_id: int, journey_data: dict):
    """Update a journey"""
    try:
        journey = db.query(Journey).filter(Journey.id == journey_id).first()
        if journey:
            # Convert route_edges to JSON string if it's a list
            if 'route_edges' in journey_data and isinstance(journey_data['route_edges'], list):
                journey_data['route_edges'] = json.dumps(journey_data['route_edges'])
            
            for key, value in journey_data.items():
                setattr(journey, key, value)
            db.commit()
            db.refresh(journey)
        return journey
    except Exception as e:
        print(f"Error updating journey: {e}")
        db.rollback()
        raise

def get_recent_journeys(db: Session, limit: int = 20):
    """Get recent journeys ordered by journey number (descending)"""
    try:
        journeys = db.query(Journey).order_by(Journey.journey_number.desc()).limit(limit).all()
        return [journey.to_dict() for journey in journeys]
    except Exception as e:
        print(f"Error getting recent journeys: {e}")
        return []

def get_journey_count(db: Session):
    """Get total number of journeys"""
    try:
        return db.query(Journey).count()
    except Exception as e:
        print(f"Error getting journey count: {e}")
        return 0

def delete_journey(db: Session, journey_id: int):
    """Delete a journey"""
    try:
        journey = db.query(Journey).filter(Journey.id == journey_id).first()
        if journey:
            db.delete(journey)
            db.commit()
        return journey
    except Exception as e:
        print(f"Error deleting journey: {e}")
        db.rollback()
        raise

def delete_last_journey(db: Session):
    """Delete the last journey (highest journey number)"""
    try:
        last_journey = db.query(Journey).order_by(Journey.journey_number.desc()).first()
        if last_journey:
            db.delete(last_journey)
            db.commit()
            return last_journey
        return None
    except Exception as e:
        print(f"Error deleting last journey: {e}")
        db.rollback()
        raise

def clear_journeys(db: Session):
    """Clear all journeys from database"""
    try:
        db.query(Journey).delete()
        db.commit()
    except Exception as e:
        print(f"Error clearing journeys: {e}")
        db.rollback()
        raise

def get_journey_statistics(db: Session):
    """Get journey statistics including MAE, RMSE, MAPE"""
    try:
        import math
        
        # Get all finished journeys
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration.isnot(None),
            Journey.predicted_eta.isnot(None)
        ).all()
        
        if not finished_journeys:
            return {
                "total_journeys": 0,
                "average_duration": 0,
                "average_distance": 0,
                "mae": 0,
                "rmse": 0,
                "mape": 0
            }
        
        # Calculate basic statistics
        total_journeys = len(finished_journeys)
        total_duration = sum(j.actual_duration for j in finished_journeys)
        total_distance = sum(j.distance for j in finished_journeys)
        average_duration = total_duration / total_journeys if total_journeys > 0 else 0
        average_distance = total_distance / total_journeys if total_journeys > 0 else 0
        
        # Calculate prediction accuracy metrics
        errors = []
        absolute_percentage_errors = []
        
        for journey in finished_journeys:
            # Calculate predicted duration from ETA
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            
            # Calculate error
            error = predicted_duration - actual_duration
            errors.append(error)
            
            # Calculate absolute percentage error (avoid division by zero)
            if actual_duration > 0:
                ape = abs(error) / actual_duration * 100
                absolute_percentage_errors.append(ape)
        
        # Calculate MAE (Mean Absolute Error)
        mae = sum(abs(e) for e in errors) / len(errors) if errors else 0
        
        # Calculate RMSE (Root Mean Square Error)
        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = sum(absolute_percentage_errors) / len(absolute_percentage_errors) if absolute_percentage_errors else 0
        
        # Calculate per-bin MAE results by duration
        short_trips_duration = []
        medium_trips_duration = []
        long_trips_duration = []
        
        # Calculate per-bin MAE results by distance
        short_trips_distance = []
        medium_trips_distance = []
        long_trips_distance = []
        
        for journey in finished_journeys:
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            distance_km = journey.distance / 1000  # Convert meters to kilometers
            error = predicted_duration - actual_duration
            
            # Duration-based bins
            if actual_duration < 278:
                short_trips_duration.append(error)
            elif actual_duration <= 609:
                medium_trips_duration.append(error)
            else:
                long_trips_duration.append(error)
            
            # Distance-based bins
            if distance_km < 4:
                short_trips_distance.append(error)
            elif distance_km <= 11:
                medium_trips_distance.append(error)
            else:
                long_trips_distance.append(error)
        
        # Calculate MAE for each duration bin
        short_mae_duration = sum(abs(e) for e in short_trips_duration) / len(short_trips_duration) if short_trips_duration else 0
        medium_mae_duration = sum(abs(e) for e in medium_trips_duration) / len(medium_trips_duration) if medium_trips_duration else 0
        long_mae_duration = sum(abs(e) for e in long_trips_duration) / len(long_trips_duration) if long_trips_duration else 0
        
        # Calculate MAE for each distance bin
        short_mae_distance = sum(abs(e) for e in short_trips_distance) / len(short_trips_distance) if short_trips_distance else 0
        medium_mae_distance = sum(abs(e) for e in medium_trips_distance) / len(medium_trips_distance) if medium_trips_distance else 0
        long_mae_distance = sum(abs(e) for e in long_trips_distance) / len(long_trips_distance) if long_trips_distance else 0
        
        return {
            "total_journeys": total_journeys,
            "average_duration": round(average_duration, 2),
            "average_distance": round(average_distance, 2),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "short_trips": {
                "mae": round(short_mae_duration, 2),
                "count": len(short_trips_duration)
            },
            "medium_trips": {
                "mae": round(medium_mae_duration, 2),
                "count": len(medium_trips_duration)
            },
            "long_trips": {
                "mae": round(long_mae_duration, 2),
                "count": len(long_trips_duration)
            },
            "short_trips_distance": {
                "mae": round(short_mae_distance, 2),
                "count": len(short_trips_distance)
            },
            "medium_trips_distance": {
                "mae": round(medium_mae_distance, 2),
                "count": len(medium_trips_distance)
            },
            "long_trips_distance": {
                "mae": round(long_mae_distance, 2),
                "count": len(long_trips_distance)
            }
        }
        
    except Exception as e:
        print(f"Error calculating journey statistics: {e}")
        return {
            "total_journeys": 0,
            "average_duration": 0,
            "average_distance": 0,
            "mae": 0,
            "rmse": 0,
            "mape": 0,
            "short_trips": {
                "mae": 0,
                "count": 0
            },
            "medium_trips": {
                "mae": 0,
                "count": 0
            },
            "long_trips": {
                "mae": 0,
                "count": 0
            },
            "short_trips_distance": {
                "mae": 0,
                "count": 0
            },
            "medium_trips_distance": {
                "mae": 0,
                "count": 0
            },
            "long_trips_distance": {
                "mae": 0,
                "count": 0
            }
        }

def get_duration_vs_mae_plot_data(db: Session):
    """Get data for Trip Duration vs MAE scatter plot"""
    try:
        # Get all finished journeys with their data
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration.isnot(None),
            Journey.predicted_eta.isnot(None)
        ).all()
        
        if not finished_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "Trip Duration (seconds)",
                "y_axis": "MAE (seconds)",
                "title": "Trip Duration vs MAE Scatter Plot"
            }
        
        data_points = []
        
        for journey in finished_journeys:
            # Calculate predicted duration from ETA
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            
            # Calculate MAE (absolute error)
            mae = abs(predicted_duration - actual_duration)
            
            # Determine trip category
            if actual_duration < 278:
                category = 'short'
            elif actual_duration <= 609:
                category = 'medium'
            else:
                category = 'long'
            
            data_points.append({
                "x": actual_duration,  # Trip duration in seconds
                "y": mae,  # MAE in seconds
                "category": category,
                "journey_id": journey.vehicle_id,
                "distance": journey.distance,
                "predicted_duration": predicted_duration,
                "actual_duration": actual_duration
            })
        
        return {
            "data_points": data_points,
            "total_journeys": len(data_points),
            "x_axis": "Trip Duration (seconds)",
            "y_axis": "MAE (seconds)",
            "title": "Trip Duration vs MAE Scatter Plot",
            "categories": {
                "short": {"label": "Short Trips (< 278s)", "color": "#3b82f6"},
                "medium": {"label": "Medium Trips (278-609s)", "color": "#f59e0b"},
                "long": {"label": "Long Trips (> 609s)", "color": "#10b981"}
            }
        }
        
    except Exception as e:
        print(f"Error getting duration vs MAE plot data: {e}")
        return {
            "data_points": [],
            "total_journeys": 0,
            "x_axis": "Trip Duration (seconds)",
            "y_axis": "MAE (seconds)",
            "title": "Trip Duration vs MAE Scatter Plot",
            "categories": {}
        }

def get_distance_vs_mae_plot_data(db: Session):
    """Get data for Trip Distance vs MAE scatter plot"""
    try:
        # Get all finished journeys with their data
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration.isnot(None),
            Journey.predicted_eta.isnot(None)
        ).all()
        
        if not finished_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "Trip Distance (meters)",
                "y_axis": "MAE (seconds)",
                "title": "Trip Distance vs MAE Scatter Plot"
            }
        
        data_points = []
        
        for journey in finished_journeys:
            # Calculate predicted duration from ETA
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            
            # Calculate MAE (absolute error)
            mae = abs(predicted_duration - actual_duration)
            
            # Determine trip category based on distance
            distance_km = journey.distance / 1000  # Convert meters to kilometers
            if distance_km < 4:
                category = 'short'
            elif distance_km <= 11:
                category = 'medium'
            else:
                category = 'long'
            
            data_points.append({
                "x": journey.distance,  # Trip distance in meters
                "y": mae,  # MAE in seconds
                "category": category,
                "journey_id": journey.vehicle_id,
                "distance": journey.distance,
                "predicted_duration": predicted_duration,
                "actual_duration": actual_duration
            })
        
        return {
            "data_points": data_points,
            "total_journeys": len(data_points),
            "x_axis": "Trip Distance (meters)",
            "y_axis": "MAE (seconds)",
            "title": "Trip Distance vs MAE Scatter Plot",
            "categories": {
                "short": {"label": "Short Trips (< 4km)", "color": "#3b82f6"},
                "medium": {"label": "Medium Trips (4-11km)", "color": "#f59e0b"},
                "long": {"label": "Long Trips (> 11km)", "color": "#10b981"}
            }
        }
        
    except Exception as e:
        print(f"Error getting distance vs MAE plot data: {e}")
        return {
            "data_points": [],
            "total_journeys": 0,
            "x_axis": "Trip Distance (meters)",
            "y_axis": "MAE (seconds)",
            "title": "Trip Distance vs MAE Scatter Plot",
            "categories": {}
        }

def get_mae_by_time_plot_data(db: Session):
    """Get data for MAE by Time of Day bar chart"""
    try:
        # Get all finished journeys with their data
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration > 0,
            Journey.predicted_eta > 0
        ).all()
        
        if not finished_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "Hour of Day",
                "y_axis": "MAE (seconds)",
                "title": "MAE by Time of Day"
            }
        
        # Group MAE by hour of day (0-23)
        hourly_mae = {}
        hourly_counts = {}
        
        for journey in finished_journeys:
            # Calculate predicted duration from ETA
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            
            # Calculate MAE (absolute error)
            mae = abs(predicted_duration - actual_duration)
            
            # Get hour of day from start_time (simulation step)
            # Convert simulation step to hour (simulation starts at midnight = 0 seconds)
            hour = journey.start_time // 3600
            
            # Initialize hour data if not exists
            if hour not in hourly_mae:
                hourly_mae[hour] = 0
                hourly_counts[hour] = 0
            
            # Add MAE to hour total
            hourly_mae[hour] += mae
            hourly_counts[hour] += 1
        
        # Calculate average MAE for each hour
        data_points = []
        for hour in range(24):
            if hour in hourly_mae and hourly_counts[hour] > 0:
                avg_mae = hourly_mae[hour] / hourly_counts[hour]
                data_points.append({
                    "hour": hour,
                    "mae": avg_mae,
                    "count": hourly_counts[hour],
                    "label": f"{hour:02d}:00"
                })
            else:
                # No data for this hour
                data_points.append({
                    "hour": hour,
                    "mae": 0,
                    "count": 0,
                    "label": f"{hour:02d}:00"
                })
        
        return {
            "data_points": data_points,
            "total_journeys": len(finished_journeys),
            "x_axis": "Hour of Day",
            "y_axis": "MAE (seconds)",
            "title": "MAE by Time of Day"
        }
        
    except Exception as e:
        print(f"Error getting MAE by time plot data: {e}")
        return {
            "data_points": [],
            "total_journeys": 0,
            "x_axis": "Hour of Day",
            "y_axis": "MAE (seconds)",
            "title": "MAE by Time of Day"
        }

def get_duration_histogram_plot_data(db: Session, category: str = 'all'):
    """Get data for Trip Duration vs MAE Histogram"""
    try:
        # Get all finished journeys with their data
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration > 0,
            Journey.predicted_eta > 0
        ).all()
        
        if not finished_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "MAE (seconds)",
                "y_axis": "Frequency",
                "title": f"Trip Duration vs MAE Histogram - {category.title()}",
                "category": category,
                "bins": []
            }
        
        # Filter by category if specified
        if category != 'all':
            if category == 'short':
                filtered_journeys = [j for j in finished_journeys if j.actual_duration < 278]
            elif category == 'medium':
                filtered_journeys = [j for j in finished_journeys if 278 <= j.actual_duration <= 609]
            elif category == 'long':
                filtered_journeys = [j for j in finished_journeys if j.actual_duration > 609]
            else:
                filtered_journeys = finished_journeys
        else:
            filtered_journeys = finished_journeys
        
        if not filtered_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "MAE (seconds)",
                "y_axis": "Frequency",
                "title": f"Trip Duration vs MAE Histogram - {category.title()}",
                "category": category,
                "bins": []
            }
        
        # Calculate MAE for each journey
        mae_values = []
        for journey in filtered_journeys:
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            mae = abs(predicted_duration - actual_duration)
            mae_values.append(mae)
        
        # Create histogram bins
        import numpy as np
        
        # Define bin edges (0 to max MAE, with 20 bins)
        max_mae = max(mae_values) if mae_values else 100
        bin_edges = np.linspace(0, max_mae, 21)  # 20 bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate histogram
        hist, _ = np.histogram(mae_values, bins=bin_edges)
        
        # Convert to data points
        data_points = []
        for i, (center, count) in enumerate(zip(bin_centers, hist)):
            data_points.append({
                "bin_center": float(center),
                "count": int(count),
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "label": f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}s"
            })
        
        return {
            "data_points": data_points,
            "total_journeys": len(filtered_journeys),
            "x_axis": "MAE (seconds)",
            "y_axis": "Frequency",
            "title": f"Trip Duration vs MAE Histogram - {category.title()}",
            "category": category,
            "bins": bin_edges.tolist(),
            "mae_values": mae_values  # For debugging
        }
        
    except Exception as e:
        print(f"Error getting duration histogram plot data: {e}")
        return {
            "data_points": [],
            "total_journeys": 0,
            "x_axis": "MAE (seconds)",
            "y_axis": "Frequency",
            "title": f"Trip Duration vs MAE Histogram - {category.title()}",
            "category": category,
            "bins": []
        }

def get_distance_histogram_plot_data(db: Session, category: str = 'all'):
    """Get data for Trip Distance vs MAE Histogram"""
    try:
        # Get all finished journeys with their data
        finished_journeys = db.query(Journey).filter(
            Journey.status == 'finished',
            Journey.actual_duration > 0,
            Journey.predicted_eta > 0
        ).all()
        
        if not finished_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "MAE (seconds)",
                "y_axis": "Frequency",
                "title": f"Trip Distance vs MAE Histogram - {category.title()}",
                "category": category,
                "bins": []
            }
        
        # Filter by category if specified (based on distance in meters)
        if category != 'all':
            if category == 'short':
                filtered_journeys = [j for j in finished_journeys if j.distance < 4000]  # < 4km
            elif category == 'medium':
                filtered_journeys = [j for j in finished_journeys if 4000 <= j.distance <= 11000]  # 4-11km
            elif category == 'long':
                filtered_journeys = [j for j in finished_journeys if j.distance > 11000]  # > 11km
            else:
                filtered_journeys = finished_journeys
        else:
            filtered_journeys = finished_journeys
        
        if not filtered_journeys:
            return {
                "data_points": [],
                "total_journeys": 0,
                "x_axis": "MAE (seconds)",
                "y_axis": "Frequency",
                "title": f"Trip Distance vs MAE Histogram - {category.title()}",
                "category": category,
                "bins": []
            }
        
        # Calculate MAE for each journey
        mae_values = []
        for journey in filtered_journeys:
            predicted_duration = journey.predicted_eta - journey.start_time
            actual_duration = journey.actual_duration
            mae = abs(predicted_duration - actual_duration)
            mae_values.append(mae)
        
        # Create histogram bins
        import numpy as np
        
        # Define bin edges (0 to max MAE, with 20 bins)
        max_mae = max(mae_values) if mae_values else 100
        bin_edges = np.linspace(0, max_mae, 21)  # 20 bins
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate histogram
        hist, _ = np.histogram(mae_values, bins=bin_edges)
        
        # Convert to data points
        data_points = []
        for i, (center, count) in enumerate(zip(bin_centers, hist)):
            data_points.append({
                "bin_center": float(center),
                "count": int(count),
                "bin_start": float(bin_edges[i]),
                "bin_end": float(bin_edges[i + 1]),
                "label": f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}s"
            })
        
        return {
            "data_points": data_points,
            "total_journeys": len(filtered_journeys),
            "x_axis": "MAE (seconds)",
            "y_axis": "Frequency",
            "title": f"Trip Distance vs MAE Histogram - {category.title()}",
            "category": category,
            "bins": bin_edges.tolist(),
            "mae_values": mae_values  # For debugging
        }
        
    except Exception as e:
        print(f"Error getting distance histogram plot data: {e}")
        return {
            "data_points": [],
            "total_journeys": 0,
            "x_axis": "MAE (seconds)",
            "y_axis": "Frequency",
            "title": f"Trip Distance vs MAE Histogram - {category.title()}",
            "category": category,
            "bins": []
        }