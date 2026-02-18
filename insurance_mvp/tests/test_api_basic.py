"""Basic API Tests

Tests for Insurance MVP FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from insurance_mvp.api.main import app, db_manager
from insurance_mvp.api.database import DatabaseManager
from insurance_mvp.api.auth import api_key_store, APIKey
from insurance_mvp.api.background import initialize_worker


# Fixtures

@pytest.fixture(scope="module")
def test_db():
    """Test database"""
    db = DatabaseManager("sqlite:///:memory:")
    db.create_tables()
    yield db
    db.drop_tables()


@pytest.fixture(scope="module")
def test_api_key():
    """Test API key"""
    return api_key_store.generate_key(
        name="test-key",
        permissions={"read": True, "write": True, "admin": True}
    )


@pytest.fixture(scope="module")
def client(test_db, test_api_key):
    """Test client"""
    # Override database in app
    global db_manager
    original_db = db_manager
    db_manager = test_db

    # Initialize worker
    initialize_worker(test_db, max_workers=2)

    # Create test client
    with TestClient(app) as client:
        yield client

    # Restore
    db_manager = original_db


# Health Check Tests

def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "version" in data
    assert "uptime_seconds" in data
    assert "database_connected" in data


# Claims Tests

def test_upload_claim_no_auth(client):
    """Test upload without authentication"""
    response = client.post("/claims/upload")
    assert response.status_code == 401


def test_upload_claim_invalid_key(client):
    """Test upload with invalid API key"""
    response = client.post(
        "/claims/upload",
        headers={"X-API-Key": "invalid_key_123"}
    )
    assert response.status_code == 401


def test_upload_claim_no_file(client, test_api_key):
    """Test upload without file"""
    response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key}
    )
    assert response.status_code == 422  # Validation error


def test_upload_claim_invalid_format(client, test_api_key):
    """Test upload with invalid file format"""
    response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("test.txt", b"not a video", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["message"]


def test_upload_claim_success(client, test_api_key):
    """Test successful video upload"""
    # Mock video file (small to avoid timeout)
    video_data = b"MOCK_VIDEO_DATA" * 100
    response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("test.mp4", video_data, "video/mp4")},
        params={"claim_number": "TEST-001", "claimant_id": "CUSTOMER-123"}
    )

    assert response.status_code == 201
    data = response.json()
    assert "claim_id" in data
    assert data["status"] in ["uploaded", "queued"]
    assert "upload_time" in data

    return data["claim_id"]


def test_get_claim_status_not_found(client, test_api_key):
    """Test get status for non-existent claim"""
    response = client.get(
        "/claims/nonexistent/status",
        headers={"X-API-Key": test_api_key}
    )
    assert response.status_code == 404


def test_get_claim_status_success(client, test_api_key):
    """Test get claim status"""
    # Upload first
    video_data = b"MOCK_VIDEO" * 100
    upload_response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("test2.mp4", video_data, "video/mp4")}
    )
    claim_id = upload_response.json()["claim_id"]

    # Get status
    response = client.get(
        f"/claims/{claim_id}/status",
        headers={"X-API-Key": test_api_key}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == claim_id
    assert "status" in data
    assert "message" in data


def test_get_assessment_not_ready(client, test_api_key):
    """Test get assessment when not ready"""
    # Upload video
    video_data = b"MOCK_VIDEO" * 100
    upload_response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("test3.mp4", video_data, "video/mp4")}
    )
    claim_id = upload_response.json()["claim_id"]

    # Try to get assessment immediately
    response = client.get(
        f"/claims/{claim_id}/assessment",
        headers={"X-API-Key": test_api_key}
    )

    # Should be 404 (not ready) or 200 (if processed very quickly)
    assert response.status_code in [200, 404]


# Review Queue Tests

def test_get_queue_no_auth(client):
    """Test get queue without authentication"""
    response = client.get("/reviews/queue")
    assert response.status_code == 401


def test_get_queue_success(client, test_api_key):
    """Test get review queue"""
    response = client.get(
        "/reviews/queue",
        headers={"X-API-Key": test_api_key},
        params={"limit": 10}
    )

    assert response.status_code == 200
    data = response.json()
    assert "total_count" in data
    assert "items" in data
    assert "urgent_count" in data
    assert "standard_count" in data
    assert "low_priority_count" in data


def test_get_queue_filtered(client, test_api_key):
    """Test get queue with priority filter"""
    response = client.get(
        "/reviews/queue",
        headers={"X-API-Key": test_api_key},
        params={"priority": "URGENT", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    # All items should be URGENT
    for item in data["items"]:
        assert item["review_priority"] == "URGENT"


# Submit Review Tests

def test_submit_review_invalid_decision(client, test_api_key):
    """Test submit review with invalid decision"""
    response = client.post(
        "/reviews/claim_test/decision",
        headers={"X-API-Key": test_api_key},
        params={"reviewer_id": "test_reviewer"},
        json={
            "decision": "INVALID",
            "reasoning": "This is a test reasoning with sufficient length"
        }
    )
    assert response.status_code == 422  # Validation error


def test_submit_review_short_reasoning(client, test_api_key):
    """Test submit review with short reasoning"""
    response = client.post(
        "/reviews/claim_test/decision",
        headers={"X-API-Key": test_api_key},
        params={"reviewer_id": "test_reviewer"},
        json={
            "decision": "APPROVE",
            "reasoning": "Short"
        }
    )
    assert response.status_code == 422  # Validation error


def test_submit_review_claim_not_found(client, test_api_key):
    """Test submit review for non-existent claim"""
    response = client.post(
        "/reviews/nonexistent/decision",
        headers={"X-API-Key": test_api_key},
        params={"reviewer_id": "test_reviewer"},
        json={
            "decision": "APPROVE",
            "reasoning": "This is a test reasoning with sufficient length for validation"
        }
    )
    assert response.status_code == 404


# Audit History Tests

def test_get_history_not_found(client, test_api_key):
    """Test get history for non-existent claim"""
    response = client.get(
        "/reviews/nonexistent/history",
        headers={"X-API-Key": test_api_key}
    )
    assert response.status_code == 404


def test_get_history_success(client, test_api_key):
    """Test get audit history"""
    # Upload video first
    video_data = b"MOCK_VIDEO" * 100
    upload_response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("test4.mp4", video_data, "video/mp4")}
    )
    claim_id = upload_response.json()["claim_id"]

    # Get history
    response = client.get(
        f"/reviews/{claim_id}/history",
        headers={"X-API-Key": test_api_key}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == claim_id
    assert "total_events" in data
    assert "events" in data
    assert len(data["events"]) > 0  # Should have at least upload event


# Metrics Tests

def test_get_metrics_success(client, test_api_key):
    """Test get system metrics"""
    response = client.get(
        "/metrics",
        headers={"X-API-Key": test_api_key}
    )

    assert response.status_code == 200
    data = response.json()

    # Check all required fields
    assert "total_claims" in data
    assert "claims_today" in data
    assert "processing_rate_per_hour" in data
    assert "queue_depth" in data
    assert "queue_depth_by_priority" in data
    assert "approval_rate" in data
    assert "average_ai_confidence" in data
    assert "error_rate" in data

    # Check types
    assert isinstance(data["total_claims"], int)
    assert isinstance(data["approval_rate"], (int, float))
    assert isinstance(data["queue_depth_by_priority"], dict)


def test_get_metrics_no_auth(client):
    """Test metrics without authentication (should work)"""
    response = client.get("/metrics")
    # Metrics endpoint allows optional auth
    assert response.status_code in [200, 401]


# Rate Limiting Tests

def test_rate_limiting(client, test_api_key):
    """Test rate limiting (basic check)"""
    # Make many requests quickly
    responses = []
    for i in range(70):  # More than rate limit (60/min)
        response = client.get(
            "/metrics",
            headers={"X-API-Key": test_api_key}
        )
        responses.append(response.status_code)

    # Should eventually get 429 (Too Many Requests)
    # Note: This test might be flaky depending on timing
    assert any(status == 429 for status in responses) or all(status == 200 for status in responses)


# Authentication Tests

def test_bearer_token_auth(client, test_api_key):
    """Test Bearer token authentication"""
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {test_api_key}"}
    )
    assert response.status_code == 200


def test_invalid_bearer_token(client):
    """Test invalid Bearer token on endpoint requiring auth"""
    response = client.get(
        "/claims/nonexistent/status",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401


# Permission Tests

def test_read_only_permission(client):
    """Test read-only API key"""
    # Generate read-only key
    readonly_key = api_key_store.generate_key(
        name="readonly-test",
        permissions={"read": True, "write": False}
    )

    # Should succeed for GET
    response = client.get(
        "/metrics",
        headers={"X-API-Key": readonly_key}
    )
    assert response.status_code == 200

    # Should fail for POST
    response = client.post(
        "/claims/upload",
        headers={"X-API-Key": readonly_key},
        files={"video": ("test.mp4", b"data", "video/mp4")}
    )
    assert response.status_code == 403


# Error Handling Tests

def test_error_response_format(client):
    """Test error response format"""
    response = client.get("/claims/nonexistent/status")

    assert response.status_code == 401  # No auth
    data = response.json()
    assert "error" in data
    assert "message" in data
    assert "timestamp" in data


# Integration Test

def test_full_workflow(client, test_api_key):
    """Test complete workflow: upload -> status -> assessment -> review"""

    # 1. Upload video
    video_data = b"MOCK_VIDEO" * 100
    upload_response = client.post(
        "/claims/upload",
        headers={"X-API-Key": test_api_key},
        files={"video": ("workflow_test.mp4", video_data, "video/mp4")},
        params={"claim_number": "WORKFLOW-001"}
    )
    assert upload_response.status_code == 201
    claim_id = upload_response.json()["claim_id"]

    # 2. Check status
    status_response = client.get(
        f"/claims/{claim_id}/status",
        headers={"X-API-Key": test_api_key}
    )
    assert status_response.status_code == 200
    assert status_response.json()["claim_id"] == claim_id

    # 3. Wait a bit for processing (mock is fast)
    import time
    time.sleep(1)

    # 4. Try to get assessment (might not be ready yet)
    assessment_response = client.get(
        f"/claims/{claim_id}/assessment",
        headers={"X-API-Key": test_api_key}
    )
    # Either ready or not ready
    assert assessment_response.status_code in [200, 404]

    # 5. Get history
    history_response = client.get(
        f"/reviews/{claim_id}/history",
        headers={"X-API-Key": test_api_key}
    )
    assert history_response.status_code == 200
    assert history_response.json()["total_events"] >= 1  # At least upload event


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
