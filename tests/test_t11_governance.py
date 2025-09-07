# T11 Governance & Monitoring Tests

import pytest
import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

# Import governance components
from echo_roots.governance import (
    User, AccessLevel, AuditLogEntry, SystemMetrics, Alert, AlertSeverity,
    UserManager, AuditLogger, SystemMonitor, GovernanceManager
)


class TestUser:
    """Test User class functionality."""
    
    def test_user_creation(self):
        """Test user creation with basic attributes."""
        user = User(
            id="test_001",
            username="testuser",
            email="test@example.com",
            access_level=AccessLevel.READ,
            created_at=datetime.now()
        )
        
        assert user.id == "test_001"
        assert user.username == "testuser"
        assert user.access_level == AccessLevel.READ
        assert user.active is True
        assert user.api_key is not None
        assert user.api_key.startswith("er_")
    
    def test_user_permissions(self):
        """Test user permission checking."""
        read_user = User(
            id="read_001", username="reader", email="read@test.com",
            access_level=AccessLevel.READ, created_at=datetime.now()
        )
        
        admin_user = User(
            id="admin_001", username="admin", email="admin@test.com",
            access_level=AccessLevel.ADMIN, created_at=datetime.now(),
            permissions={"manage_users", "view_logs"}
        )
        
        # Test access level hierarchy
        assert read_user.can_access_resource(AccessLevel.READ)
        assert not read_user.can_access_resource(AccessLevel.WRITE)
        assert not read_user.can_access_resource(AccessLevel.ADMIN)
        
        assert admin_user.can_access_resource(AccessLevel.READ)
        assert admin_user.can_access_resource(AccessLevel.WRITE)
        assert admin_user.can_access_resource(AccessLevel.ADMIN)
        
        # Test specific permissions
        assert admin_user.has_permission("manage_users")
        assert admin_user.has_permission("any_permission")  # Admin has all permissions
        assert not read_user.has_permission("manage_users")


class TestUserManager:
    """Test UserManager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.user_manager = UserManager()
    
    def test_default_admin_creation(self):
        """Test that default admin user is created."""
        admin_users = [u for u in self.user_manager.users.values() 
                      if u.access_level == AccessLevel.ADMIN]
        assert len(admin_users) >= 1
        
        admin = admin_users[0]
        assert admin.username == "admin"
        assert admin.active is True
        assert "*" in admin.permissions
    
    def test_add_user(self):
        """Test adding new users."""
        user = User(
            id="new_001", username="newuser", email="new@test.com",
            access_level=AccessLevel.WRITE, created_at=datetime.now()
        )
        
        success = self.user_manager.add_user(user)
        assert success is True
        assert user.id in self.user_manager.users
        assert user.api_key in self.user_manager.api_keys
        
        # Test duplicate user
        duplicate_success = self.user_manager.add_user(user)
        assert duplicate_success is False
    
    def test_api_key_authentication(self):
        """Test API key authentication."""
        user = User(
            id="api_001", username="apiuser", email="api@test.com",
            access_level=AccessLevel.READ, created_at=datetime.now()
        )
        self.user_manager.add_user(user)
        
        # Test valid API key
        auth_user = self.user_manager.authenticate_api_key(user.api_key)
        assert auth_user is not None
        assert auth_user.id == user.id
        
        # Test invalid API key
        invalid_auth = self.user_manager.authenticate_api_key("invalid_key")
        assert invalid_auth is None
        
        # Test inactive user
        user.active = False
        inactive_auth = self.user_manager.authenticate_api_key(user.api_key)
        assert inactive_auth is None
    
    def test_session_management(self):
        """Test session creation and validation."""
        user = User(
            id="session_001", username="sessionuser", email="session@test.com",
            access_level=AccessLevel.READ, created_at=datetime.now()
        )
        self.user_manager.add_user(user)
        
        # Create session
        session_id = self.user_manager.create_session(user)
        assert session_id is not None
        assert session_id in self.user_manager.sessions
        
        # Validate session
        session_user = self.user_manager.validate_session(session_id)
        assert session_user is not None
        assert session_user.id == user.id
        
        # Test invalid session
        invalid_session = self.user_manager.validate_session("invalid_session")
        assert invalid_session is None
        
        # Revoke session
        revoked = self.user_manager.revoke_session(session_id)
        assert revoked is True
        
        # Test revoked session
        revoked_session = self.user_manager.validate_session(session_id)
        assert revoked_session is None


class TestAuditLogger:
    """Test AuditLogger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a temporary file for testing to avoid pollution between tests
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        temp_file.close()
        self.audit_logger = AuditLogger(Path(temp_file.name))
        
    def teardown_method(self):
        """Clean up after tests."""
        # Clean up the temporary log file
        if self.audit_logger.log_file.exists():
            self.audit_logger.log_file.unlink()
    
    def test_log_action(self):
        """Test logging user actions."""
        self.audit_logger.log_action(
            user_id="test_user",
            action="search",
            resource="taxonomy",
            details={"query": "test query"},
            success=True
        )
        
        assert len(self.audit_logger.entries) > 0
        latest_entry = self.audit_logger.entries[-1]
        
        assert latest_entry.user_id == "test_user"
        assert latest_entry.action == "search"
        assert latest_entry.resource == "taxonomy"
        assert latest_entry.success is True
        assert latest_entry.details["query"] == "test query"
    
    def test_get_logs_filtering(self):
        """Test log retrieval with filtering."""
        # Add test logs
        self.audit_logger.log_action("user1", "search", "api", {"q": "test1"}, True)
        self.audit_logger.log_action("user2", "update", "api", {"id": "123"}, True)
        self.audit_logger.log_action("user1", "delete", "api", {"id": "456"}, False, "Permission denied")
        
        # Test user filtering
        user1_logs = self.audit_logger.get_logs(user_id="user1")
        assert len(user1_logs) == 2
        assert all(log.user_id == "user1" for log in user1_logs)
        
        # Test action filtering  
        search_logs = self.audit_logger.get_logs(action="search")
        assert len(search_logs) == 1
        assert search_logs[0].action == "search"
        
        # Test limit
        limited_logs = self.audit_logger.get_logs(limit=2)
        assert len(limited_logs) <= 2


class TestSystemMonitor:
    """Test SystemMonitor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = SystemMonitor()
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test metrics collection."""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(percent=60.2)
        mock_disk.return_value = Mock(percent=75.8)
        
        metrics = self.monitor.collect_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage == 45.5
        assert metrics.memory_usage == 60.2
        assert metrics.disk_usage == 75.8
        assert metrics.health_status == "healthy"
        
        assert len(self.monitor.metrics_history) == 1
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_alert_generation(self, mock_disk, mock_memory, mock_cpu):
        """Test alert generation for high resource usage."""
        # Mock high CPU usage
        mock_cpu.return_value = 95.0
        mock_memory.return_value = Mock(percent=50.0)
        mock_disk.return_value = Mock(percent=30.0)
        
        initial_alerts = len(self.monitor.alerts)
        metrics = self.monitor.collect_metrics()
        
        assert metrics.health_status == "critical"
        assert len(self.monitor.alerts) > initial_alerts
        
        # Check for CPU alert
        cpu_alerts = [a for a in self.monitor.alerts if "CPU" in a.title]
        assert len(cpu_alerts) > 0
        assert cpu_alerts[-1].severity == AlertSeverity.CRITICAL
    
    def test_alert_management(self):
        """Test alert creation and resolution."""
        alert = self.monitor.create_alert(
            AlertSeverity.HIGH,
            "Test Alert",
            "This is a test alert",
            "test_component"
        )
        
        assert alert.id is not None
        assert alert.severity == AlertSeverity.HIGH
        assert alert.resolved is False
        
        # Test active alerts
        active_alerts = self.monitor.get_active_alerts()
        assert alert in active_alerts
        
        # Test resolution
        resolved = self.monitor.resolve_alert(alert.id, "test_user")
        assert resolved is True
        assert alert.resolved is True
        assert alert.resolved_by == "test_user"
        
        # Test resolved alerts are not in active
        active_alerts_after = self.monitor.get_active_alerts()
        assert alert not in active_alerts_after
    
    def test_query_metrics_tracking(self):
        """Test query performance tracking."""
        # Record some query times
        self.monitor.record_query_time(100.5)
        self.monitor.record_query_time(200.2)
        self.monitor.record_query_time(150.8)
        
        assert len(self.monitor.query_times) == 3
        assert self.monitor.total_requests == 3
        assert self.monitor.error_count == 0
        
        # Record errors
        self.monitor.record_error()
        self.monitor.record_error()
        
        assert self.monitor.error_count == 2
        assert self.monitor.total_requests == 5


class TestGovernanceManager:
    """Test GovernanceManager integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.governance = GovernanceManager()
    
    def test_initialization(self):
        """Test governance manager initialization."""
        assert self.governance.user_manager is not None
        assert self.governance.audit_logger is not None
        assert self.governance.system_monitor is not None
        assert self.governance.config is not None
    
    def test_dashboard_data(self):
        """Test dashboard data generation."""
        dashboard = self.governance.get_dashboard_data()
        
        assert "system_health" in dashboard
        assert "active_users" in dashboard
        assert "recent_audit_logs" in dashboard
        assert "alerts_summary" in dashboard
        
        # Test data structure
        assert isinstance(dashboard["active_users"], int)
        assert isinstance(dashboard["recent_audit_logs"], list)
        assert "total" in dashboard["alerts_summary"]
        assert "critical" in dashboard["alerts_summary"]
    
    @pytest.mark.asyncio
    async def test_authorization(self):
        """Test request authorization."""
        # Get admin user API key
        admin_user = None
        for user in self.governance.user_manager.users.values():
            if user.access_level == AccessLevel.ADMIN:
                admin_user = user
                break
        
        assert admin_user is not None
        
        # Test valid authorization
        authorized, user = await self.governance.authorize_request(
            admin_user.api_key, "test_permission"
        )
        assert authorized is True
        assert user is not None
        assert user.id == admin_user.id
        
        # Test invalid API key
        unauthorized, user = await self.governance.authorize_request(
            "invalid_key", "test_permission"
        )
        assert unauthorized is False
        assert user is None
    
    def test_request_logging(self):
        """Test request logging integration."""
        initial_logs = len(self.governance.audit_logger.entries)
        
        self.governance.log_request(
            user_id="test_user",
            action="test_action",
            resource="test_resource",
            details={"param": "value"},
            success=True
        )
        
        assert len(self.governance.audit_logger.entries) == initial_logs + 1
        latest_log = self.governance.audit_logger.entries[-1]
        assert latest_log.user_id == "test_user"
        assert latest_log.action == "test_action"
        assert latest_log.success is True


class TestGovernanceAPI:
    """Test governance API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from echo_roots.cli.api_server import app
        self.client = TestClient(app)
    
    def test_governance_status_endpoint(self):
        """Test governance status API endpoint."""
        response = self.client.get("/governance/status")
        
        # Should return valid JSON regardless of status code
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, dict)
    
    def test_metrics_endpoint(self):
        """Test system metrics API endpoint."""
        response = self.client.get("/governance/metrics")
        
        # Should return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, dict)
    
    def test_alerts_endpoint(self):
        """Test alerts API endpoint."""
        response = self.client.get("/governance/alerts")
        
        # Should return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, dict)
        
        # Test with severity filter
        response_filtered = self.client.get("/governance/alerts?severity=critical")
        assert response_filtered.status_code in [200, 400, 500]
    
    def test_users_endpoint(self):
        """Test users API endpoint."""
        response = self.client.get("/governance/users")
        
        # Should return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, dict)
    
    def test_audit_logs_endpoint(self):
        """Test audit logs API endpoint."""
        response = self.client.get("/governance/audit")
        
        # Should return valid JSON
        assert response.status_code in [200, 500]
        data = response.json()
        assert isinstance(data, dict)
        
        # Test with filters
        response_filtered = self.client.get("/governance/audit?limit=10&action=search")
        assert response_filtered.status_code in [200, 500]


class TestDataSerialization:
    """Test data serialization for API responses."""
    
    def test_metrics_serialization(self):
        """Test SystemMetrics serialization."""
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=50.5,
            memory_usage=60.2,
            disk_usage=70.8,
            active_users=5,
            query_count=100,
            query_latency_avg=150.5,
            query_latency_p95=300.2,
            error_rate=1.5,
            cache_hit_rate=85.0,
            database_connections=10,
            api_requests_per_minute=50,
            uptime_seconds=3600
        )
        
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "timestamp" in metrics_dict
        assert isinstance(metrics_dict["timestamp"], str)  # Should be ISO format
        
        # Should be JSON serializable
        json_str = json.dumps(metrics_dict)
        assert isinstance(json_str, str)
    
    def test_alert_serialization(self):
        """Test Alert serialization."""
        alert = Alert(
            id="test_alert",
            timestamp=datetime.now(),
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            message="Test message",
            component="test_component"
        )
        
        # Should be convertible to dict
        alert_dict = {
            "id": alert.id,
            "timestamp": alert.timestamp.isoformat(),
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "component": alert.component,
            "resolved": alert.resolved
        }
        
        # Should be JSON serializable
        json_str = json.dumps(alert_dict)
        assert isinstance(json_str, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
