from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server

# Counters

MESSAGES_RECEIVED = Counter(
    "aki_messages_received_total",
    "Total number of HL7 messages received",
    ["message_type"],  # admit, discharge, creatinine, unknown, parse_error
)

BLOOD_TESTS_RECEIVED = Counter(
    "aki_blood_tests_received_total",
    "Total number of creatinine blood test results received",
)

AKI_PREDICTIONS_TOTAL = Counter(
    "aki_predictions_total",
    "Total number of AKI model predictions",
    ["result"],  # positive, negative
)

PAGER_REQUESTS_TOTAL = Counter(
    "aki_pager_requests_total",
    "Total number of pager HTTP requests",
    ["status"],  # success, error
)

MLLP_RECONNECTIONS = Counter(
    "aki_mllp_reconnections_total",
    "Total number of reconnections to the MLLP socket",
)

MLLP_READ_ERRORS = Counter(
    "aki_mllp_read_errors_total",
    "Total number of MLLP read errors (e.g. connection reset)",
)

# Histograms
MESSAGE_PROCESSING_LATENCY = Histogram(
    "aki_message_processing_duration_seconds",
    "Time spent processing a single HL7 message",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

CREATININE_VALUE = Histogram(
    "aki_creatinine_value",
    "Distribution of creatinine blood test result values",
    buckets=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0),
)

# Gauges

MLLP_CONNECTED = Gauge(
    "aki_mllp_connected",
    "Whether the MLLP client is currently connected (1=connected, 0=disconnected)",
)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics HTTP server on the given port."""
    start_http_server(port)
