"""Example API Client

Demonstrates how to interact with the Insurance MVP API.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import requests
from requests.exceptions import RequestException


class InsuranceAPIClient:
    """
    Python client for Insurance MVP API.

    Example:
        client = InsuranceAPIClient("http://localhost:8000", "ins_your_api_key")
        claim_id = client.upload_video("dashcam.mp4", claim_number="CLM-001")
        assessment = client.wait_for_assessment(claim_id)
        print(f"Severity: {assessment['severity']}")
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        """
        Initialize API client.

        Args:
            base_url: API base URL (e.g., http://localhost:8000)
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {"X-API-Key": api_key}

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., /claims/upload)
            **kwargs: Additional arguments for requests.request()

        Returns:
            Response JSON

        Raises:
            RequestException: On HTTP errors
        """
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("headers", {}).update(self.headers)
        kwargs.setdefault("timeout", self.timeout)

        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"API Error: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"Details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"Response: {e.response.text}")
            raise

    # Claims API

    def upload_video(
        self,
        video_path: str,
        claim_number: Optional[str] = None,
        claimant_id: Optional[str] = None,
    ) -> str:
        """
        Upload dashcam video.

        Args:
            video_path: Path to video file
            claim_number: Optional claim reference number
            claimant_id: Optional claimant ID

        Returns:
            Claim ID

        Raises:
            RequestException: On upload failure
        """
        video_file = Path(video_path)
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"Uploading video: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.2f} MB)")

        with open(video_file, "rb") as f:
            files = {"video": (video_file.name, f, "video/mp4")}
            params = {}
            if claim_number:
                params["claim_number"] = claim_number
            if claimant_id:
                params["claimant_id"] = claimant_id

            result = self._request("POST", "/claims/upload", files=files, params=params)

        claim_id = result["claim_id"]
        print(f"✓ Uploaded successfully: {claim_id}")
        return claim_id

    def get_status(self, claim_id: str) -> Dict[str, Any]:
        """
        Get claim processing status.

        Args:
            claim_id: Claim ID

        Returns:
            Status information
        """
        return self._request("GET", f"/claims/{claim_id}/status")

    def get_assessment(self, claim_id: str) -> Dict[str, Any]:
        """
        Get AI assessment results.

        Args:
            claim_id: Claim ID

        Returns:
            Assessment data

        Raises:
            RequestException: If assessment not ready
        """
        return self._request("GET", f"/claims/{claim_id}/assessment")

    def wait_for_assessment(
        self,
        claim_id: str,
        timeout: int = 300,
        poll_interval: int = 2,
    ) -> Dict[str, Any]:
        """
        Wait for AI assessment to complete.

        Args:
            claim_id: Claim ID
            timeout: Maximum wait time in seconds
            poll_interval: Time between status checks in seconds

        Returns:
            Assessment data

        Raises:
            TimeoutError: If assessment not completed within timeout
            RequestException: On processing failure
        """
        start_time = time.time()
        last_progress = 0.0

        print(f"Waiting for assessment: {claim_id}")

        while time.time() - start_time < timeout:
            status = self.get_status(claim_id)

            # Update progress
            current_progress = status.get("progress_percent", 0.0)
            if current_progress > last_progress:
                print(f"  Progress: {current_progress:.0f}%")
                last_progress = current_progress

            # Check status
            if status["status"] == "assessed":
                print("✓ Assessment complete")
                return self.get_assessment(claim_id)
            elif status["status"] == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise RequestException(f"Processing failed: {error_msg}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Assessment timeout after {timeout}s")

    # Reviews API

    def get_queue(
        self,
        priority: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get review queue.

        Args:
            priority: Filter by priority (URGENT, STANDARD, LOW_PRIORITY)
            limit: Maximum items to return
            offset: Offset for pagination

        Returns:
            Queue data with items
        """
        params = {"limit": limit, "offset": offset}
        if priority:
            params["priority"] = priority

        return self._request("GET", "/reviews/queue", params=params)

    def submit_review(
        self,
        claim_id: str,
        reviewer_id: str,
        decision: str,
        reasoning: str,
        severity_override: Optional[str] = None,
        fault_ratio_override: Optional[float] = None,
        fraud_override: Optional[bool] = None,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit human review decision.

        Args:
            claim_id: Claim ID
            reviewer_id: Reviewer identifier
            decision: Review decision (APPROVE, REJECT, REQUEST_MORE_INFO)
            reasoning: Detailed reasoning (min 10 characters)
            severity_override: Override AI severity
            fault_ratio_override: Override fault ratio
            fraud_override: Override fraud flag
            comments: Additional comments

        Returns:
            Review response
        """
        data = {
            "decision": decision,
            "reasoning": reasoning,
        }
        if severity_override:
            data["severity_override"] = severity_override
        if fault_ratio_override is not None:
            data["fault_ratio_override"] = fault_ratio_override
        if fraud_override is not None:
            data["fraud_override"] = fraud_override
        if comments:
            data["comments"] = comments

        return self._request(
            "POST",
            f"/reviews/{claim_id}/decision",
            params={"reviewer_id": reviewer_id},
            json=data,
        )

    def get_history(self, claim_id: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get audit history for claim.

        Args:
            claim_id: Claim ID
            limit: Maximum events to return

        Returns:
            Audit history
        """
        return self._request("GET", f"/reviews/{claim_id}/history", params={"limit": limit})

    # System API

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics.

        Returns:
            Metrics data
        """
        return self._request("GET", "/metrics")

    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.

        Returns:
            Health status
        """
        # Health check doesn't require auth
        url = f"{self.base_url}/health"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()


# Example Usage

def main():
    """Example usage of Insurance API client"""

    # Configuration
    API_BASE = "http://localhost:8000"
    API_KEY = "ins_your_api_key_here"  # Replace with actual key from startup logs

    # Initialize client
    client = InsuranceAPIClient(API_BASE, API_KEY)

    # 1. Health Check
    print("\n=== Health Check ===")
    health = client.health_check()
    print(f"Status: {health['status']}")
    print(f"Version: {health['version']}")
    print(f"Database: {'✓' if health['database_connected'] else '✗'}")

    # 2. Upload Video (requires actual video file)
    print("\n=== Upload Video ===")
    # Uncomment and provide actual video path:
    # claim_id = client.upload_video(
    #     "path/to/dashcam.mp4",
    #     claim_number="CLM-2026-001234",
    #     claimant_id="CUSTOMER-5678",
    # )

    # For demo, use mock claim ID
    claim_id = "claim_demo_123456"
    print(f"Using demo claim ID: {claim_id}")

    # 3. Wait for Assessment
    print("\n=== Wait for Assessment ===")
    # Uncomment to wait for actual processing:
    # try:
    #     assessment = client.wait_for_assessment(claim_id, timeout=120)
    #     print(f"Severity: {assessment['severity']}")
    #     print(f"Confidence: {assessment['confidence']:.2%}")
    #     print(f"Fault Ratio: {assessment['fault_assessment']['fault_ratio']:.1f}%")
    #     print(f"Fraud Risk: {assessment['fraud_risk']['risk_score']:.2%}")
    # except Exception as e:
    #     print(f"Error: {e}")

    # 4. Get Review Queue
    print("\n=== Review Queue ===")
    try:
        queue = client.get_queue(limit=10)
        print(f"Total pending: {queue['total_count']}")
        print(f"  Urgent: {queue['urgent_count']}")
        print(f"  Standard: {queue['standard_count']}")
        print(f"  Low Priority: {queue['low_priority_count']}")

        # Show first few items
        for i, item in enumerate(queue["items"][:3], 1):
            print(f"\n  {i}. {item['claim_id']}")
            print(f"     Priority: {item['review_priority']}")
            print(f"     Severity: {item['severity']} (confidence: {item['confidence']:.2%})")
            print(f"     Fraud Risk: {item['fraud_risk_score']:.2%}")
    except Exception as e:
        print(f"Error: {e}")

    # 5. Submit Review (example)
    print("\n=== Submit Review (Example) ===")
    # Uncomment to submit actual review:
    # try:
    #     result = client.submit_review(
    #         claim_id=claim_id,
    #         reviewer_id="reviewer_alice",
    #         decision="APPROVE",
    #         reasoning="AI assessment is accurate based on video evidence. No fraud indicators.",
    #         fraud_override=False,
    #         comments="Clear case, good for training data.",
    #     )
    #     print(f"Review submitted: {result['status']}")
    # except Exception as e:
    #     print(f"Error: {e}")
    print("(Skipped - uncomment to test)")

    # 6. System Metrics
    print("\n=== System Metrics ===")
    try:
        metrics = client.get_metrics()
        print(f"Total Claims: {metrics['total_claims']}")
        print(f"Claims Today: {metrics['claims_today']}")
        print(f"Processing Rate: {metrics['processing_rate_per_hour']:.1f} claims/hour")
        print(f"Queue Depth: {metrics['queue_depth']}")
        print(f"Approval Rate: {metrics['approval_rate']:.1%}")
        print(f"Average AI Confidence: {metrics['average_ai_confidence']:.1%}")
        print(f"Average Fraud Risk: {metrics['average_fraud_risk']:.1%}")
        print(f"Error Rate: {metrics['error_rate']:.1%}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
