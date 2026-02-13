"""Evaluation framework for SOPilot neural components and VIGIL-RAG."""

from .statistical import (
    AblationStudy,
    bootstrap_confidence_interval,
    intraclass_correlation,
    permutation_test,
)
from .vigil_benchmark import (
    BenchmarkQuery,
    BenchmarkReport,
    VIGILBenchmarkRunner,
    format_report,
    report_to_dict,
)
from .vigil_metrics import (
    EventDetectionResult,
    EvidenceRecallResult,
    RetrievalResult,
    event_detection_metrics,
    evidence_recall_at_k,
    mrr,
    ndcg_at_k,
)

__all__ = [
    # Statistical evaluation
    "bootstrap_confidence_interval",
    "permutation_test",
    "intraclass_correlation",
    "AblationStudy",
    # VIGIL-RAG metrics
    "EvidenceRecallResult",
    "EventDetectionResult",
    "RetrievalResult",
    "evidence_recall_at_k",
    "event_detection_metrics",
    "mrr",
    "ndcg_at_k",
    # VIGIL-RAG benchmark
    "BenchmarkQuery",
    "BenchmarkReport",
    "VIGILBenchmarkRunner",
    "format_report",
    "report_to_dict",
]
