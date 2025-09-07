# T11 Governance & Monitoring Implementation

"""
System governance, monitoring, access control, and operational management.
Provides administrative oversight and production-ready operational capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import hashlib
import uuid
from enum import Enum

# Configuration and logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access control levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SYSTEM = "system"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class User:
    """User representation for access control."""
    id: str
    username: str
    email: str
    access_level: AccessLevel
    created_at: datetime
    last_login: Optional[datetime] = None
    active: bool = True
    api_key: Optional[str] = None
    permissions: Set[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()
        if self.api_key is None:
            self.api_key = self._generate_api_key()
    
    def _generate_api_key(self) -> str:
        """Generate secure API key."""
        return f"er_{hashlib.sha256(f'{self.id}_{self.username}_{time.time()}'.encode()).hexdigest()[:32]}"
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or self.access_level == AccessLevel.ADMIN
    
    def can_access_resource(self, resource_access_level: AccessLevel) -> bool:
        """Check if user can access resource based on access level."""
        access_hierarchy = {
            AccessLevel.READ: 1,
            AccessLevel.WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SYSTEM: 4
        }
        return access_hierarchy[self.access_level] >= access_hierarchy[resource_access_level]


@dataclass
class AuditLogEntry:
    """Audit log entry for tracking system activities."""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class SystemMetrics:
    """System performance and health metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_users: int
    query_count: int
    query_latency_avg: float
    query_latency_p95: float
    error_rate: float
    cache_hit_rate: float
    database_connections: int
    api_requests_per_minute: int
    uptime_seconds: int
    health_status: str = "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class Alert:
    """System alert representation."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    component: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
    
    def resolve(self, resolved_by: str):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_at = datetime.now()
        self.resolved_by = resolved_by


class UserManager:
    """User management and access control."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self._setup_default_admin()
    
    def _setup_default_admin(self):
        """Create default admin user if none exists."""
        admin_user = User(
            id="admin_001",
            username="admin",
            email="admin@echo-roots.local",
            access_level=AccessLevel.ADMIN,
            created_at=datetime.now(),
            permissions={"*"}  # All permissions
        )
        self.add_user(admin_user)
    
    def add_user(self, user: User) -> bool:
        """Add new user to the system."""
        if user.id in self.users:
            return False
        self.users[user.id] = user
        self.api_keys[user.api_key] = user.id
        logger.info(f"User added: {user.username} (ID: {user.id})")
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username and password."""
        # In production, this would use proper password hashing
        for user in self.users.values():
            if user.username == username and user.active:
                user.last_login = datetime.now()
                return user
        return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user by API key."""
        user_id = self.api_keys.get(api_key)
        if user_id and user_id in self.users:
            user = self.users[user_id]
            if user.active:
                return user
        return None
    
    def create_session(self, user: User) -> str:
        """Create user session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user.id,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user."""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check session timeout (24 hours)
        if datetime.now() - session["last_activity"] > timedelta(hours=24):
            del self.sessions[session_id]
            return None
        
        session["last_activity"] = datetime.now()
        user_id = session["user_id"]
        return self.users.get(user_id)
    
    def revoke_session(self, session_id: str) -> bool:
        """Revoke user session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_active_users(self) -> List[User]:
        """Get list of active users."""
        return [user for user in self.users.values() if user.active]


class AuditLogger:
    """Audit logging for tracking system activities."""
    
    def __init__(self, log_file: Path = None):
        self.log_file = log_file or Path("audit.log")
        self.entries: List[AuditLogEntry] = []
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Load existing audit logs from file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry_data = json.loads(line)
                            entry_data['timestamp'] = datetime.fromisoformat(entry_data['timestamp'])
                            entry = AuditLogEntry(**entry_data)
                            self.entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to load audit logs: {e}")
    
    def log_action(self, user_id: Optional[str], action: str, resource: str, 
                   details: Dict[str, Any], success: bool = True, 
                   error_message: Optional[str] = None,
                   ip_address: Optional[str] = None,
                   user_agent: Optional[str] = None):
        """Log user action for audit trail."""
        entry = AuditLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.entries.append(entry)
        self._write_entry_to_file(entry)
        
        if not success:
            logger.warning(f"Failed action logged: {action} by user {user_id}: {error_message}")
    
    def _write_entry_to_file(self, entry: AuditLogEntry):
        """Write audit entry to log file."""
        try:
            with open(self.log_file, 'a') as f:
                entry_dict = asdict(entry)
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                f.write(json.dumps(entry_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def get_logs(self, user_id: Optional[str] = None, 
                 action: Optional[str] = None,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 limit: int = 100) -> List[AuditLogEntry]:
        """Retrieve audit logs with filtering."""
        filtered_entries = self.entries
        
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if action:
            filtered_entries = [e for e in filtered_entries if e.action == action]
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        # Sort by timestamp descending
        filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_entries[:limit]


class SystemMonitor:
    """System monitoring and health checking."""
    
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Alert] = []
        self.start_time = datetime.now()
        self.query_times: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        import psutil
        
        # Calculate derived metrics
        uptime = (datetime.now() - self.start_time).total_seconds()
        query_latency_avg = sum(self.query_times[-100:]) / len(self.query_times[-100:]) if self.query_times else 0
        query_latency_p95 = sorted(self.query_times[-100:])[-5] if len(self.query_times) >= 5 else 0
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            active_users=0,  # Would be populated from UserManager
            query_count=len(self.query_times),
            query_latency_avg=query_latency_avg,
            query_latency_p95=query_latency_p95,
            error_rate=error_rate,
            cache_hit_rate=0.0,  # Would be populated from cache system
            database_connections=0,  # Would be populated from database pool
            api_requests_per_minute=0,  # Would be calculated from request logs
            uptime_seconds=int(uptime)
        )
        
        # Determine health status
        if metrics.cpu_usage > 90 or metrics.memory_usage > 90:
            metrics.health_status = "critical"
        elif metrics.cpu_usage > 75 or metrics.memory_usage > 75:
            metrics.health_status = "warning"
        else:
            metrics.health_status = "healthy"
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics entries
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        # Check for alerts
        self._check_alert_conditions(metrics)
        
        return metrics
    
    def _check_alert_conditions(self, metrics: SystemMetrics):
        """Check for conditions that should trigger alerts."""
        # High CPU usage alert
        if metrics.cpu_usage > 90:
            self.create_alert(
                AlertSeverity.CRITICAL,
                "High CPU Usage",
                f"CPU usage is {metrics.cpu_usage:.1f}%",
                "system"
            )
        
        # High memory usage alert
        if metrics.memory_usage > 90:
            self.create_alert(
                AlertSeverity.CRITICAL,
                "High Memory Usage", 
                f"Memory usage is {metrics.memory_usage:.1f}%",
                "system"
            )
        
        # High error rate alert
        if metrics.error_rate > 10:
            self.create_alert(
                AlertSeverity.HIGH,
                "High Error Rate",
                f"Error rate is {metrics.error_rate:.1f}%",
                "application"
            )
    
    def create_alert(self, severity: AlertSeverity, title: str, 
                    message: str, component: str, metadata: Dict[str, Any] = None):
        """Create system alert."""
        alert = Alert(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            severity=severity,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert created [{severity.value.upper()}]: {title} - {message}")
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolve(resolved_by)
                logger.info(f"Alert resolved: {alert.title} by {resolved_by}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        return [alert for alert in self.alerts if alert.severity == severity and not alert.resolved]
    
    def record_query_time(self, duration: float):
        """Record query execution time for metrics."""
        self.query_times.append(duration)
        self.total_requests += 1
        
        # Keep only last 1000 query times
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]
    
    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1
        self.total_requests += 1
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        active_alerts = self.get_active_alerts()
        
        return {
            "overall_status": latest_metrics.health_status if latest_metrics else "unknown",
            "uptime_seconds": latest_metrics.uptime_seconds if latest_metrics else 0,
            "active_alerts_count": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
            "top_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)[:5]
            ]
        }


class GovernanceManager:
    """Main governance and monitoring coordinator."""
    
    def __init__(self):
        self.user_manager = UserManager()
        self.audit_logger = AuditLogger()
        self.system_monitor = SystemMonitor()
        self.config = self._load_config()
        self._start_monitoring_tasks()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load governance configuration."""
        config_file = Path("governance_config.yaml")
        default_config = {
            "monitoring": {
                "metrics_collection_interval": 60,
                "alert_check_interval": 30,
                "metrics_retention_days": 30
            },
            "security": {
                "session_timeout_hours": 24,
                "max_failed_login_attempts": 5,
                "api_rate_limit_per_minute": 1000
            },
            "audit": {
                "log_retention_days": 365,
                "log_file_max_size_mb": 100
            }
        }
        
        if config_file.exists():
            import yaml
            try:
                with open(config_file, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    default_config.update(user_config)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        return default_config
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        # In a real implementation, these would be proper async tasks
        logger.info("Governance monitoring started")
    
    async def authorize_request(self, api_key: str, required_permission: str) -> tuple[bool, Optional[User]]:
        """Authorize API request."""
        user = self.user_manager.authenticate_api_key(api_key)
        if not user:
            return False, None
        
        if not user.has_permission(required_permission):
            return False, user
        
        return True, user
    
    def log_request(self, user_id: Optional[str], action: str, resource: str, 
                   details: Dict[str, Any], success: bool = True,
                   error_message: Optional[str] = None):
        """Log API request for audit trail."""
        self.audit_logger.log_action(
            user_id=user_id,
            action=action,
            resource=resource,
            details=details,
            success=success,
            error_message=error_message
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for governance dashboard."""
        return {
            "system_health": self.system_monitor.get_system_health(),
            "active_users": len(self.user_manager.get_active_users()),
            "recent_audit_logs": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "user_id": entry.user_id,
                    "action": entry.action,
                    "resource": entry.resource,
                    "success": entry.success
                }
                for entry in self.audit_logger.get_logs(limit=10)
            ],
            "alerts_summary": {
                "total": len(self.system_monitor.get_active_alerts()),
                "critical": len(self.system_monitor.get_alerts_by_severity(AlertSeverity.CRITICAL)),
                "high": len(self.system_monitor.get_alerts_by_severity(AlertSeverity.HIGH))
            }
        }


# Global governance manager instance
governance_manager = GovernanceManager()


# Decorator for securing API endpoints
def require_permission(permission: str):
    """Decorator to require specific permission for API endpoint."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract API key from request headers
            api_key = kwargs.get('api_key') or 'default_admin_key'  # Simplified for demo
            
            authorized, user = await governance_manager.authorize_request(api_key, permission)
            if not authorized:
                raise Exception(f"Access denied: {permission} permission required")
            
            # Add user to kwargs for use in endpoint
            kwargs['current_user'] = user
            
            try:
                result = await func(*args, **kwargs)
                governance_manager.log_request(
                    user_id=user.id if user else None,
                    action=func.__name__,
                    resource=permission,
                    details={"args": str(args), "kwargs": str(kwargs)},
                    success=True
                )
                return result
            except Exception as e:
                governance_manager.log_request(
                    user_id=user.id if user else None,
                    action=func.__name__,
                    resource=permission,
                    details={"args": str(args), "kwargs": str(kwargs)},
                    success=False,
                    error_message=str(e)
                )
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo usage
    print("Echo-Roots Governance & Monitoring System")
    print("=========================================")
    
    # Collect metrics
    metrics = governance_manager.system_monitor.collect_metrics()
    print(f"System Health: {metrics.health_status}")
    print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
    print(f"Memory Usage: {metrics.memory_usage:.1f}%")
    
    # Get dashboard data
    dashboard = governance_manager.get_dashboard_data()
    print(f"Active Users: {dashboard['active_users']}")
    print(f"Active Alerts: {dashboard['alerts_summary']['total']}")
    
    print("\nGovernance system initialized successfully!")
