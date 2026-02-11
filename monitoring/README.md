# SOPilot Monitoring Setup

This directory contains Prometheus and Grafana configurations for production monitoring.

## Quick Start

### Prerequisites

- Prometheus installed in your Kubernetes cluster
- Grafana installed and configured
- Prometheus Operator (for PrometheusRule CRDs)

### Deploy Monitoring

```bash
# Apply Prometheus alerting rules
kubectl apply -f monitoring/prometheus-rules.yaml

# Import Grafana dashboard
kubectl create configmap sopilot-grafana-dashboard \
  --from-file=monitoring/grafana-dashboard.json \
  -n monitoring

# Or import manually via Grafana UI:
# Dashboard > Import > Upload JSON file > monitoring/grafana-dashboard.json
```

## Grafana Dashboard

### Features

1. **Job Processing Rate** - Real-time job throughput (ingest, score, training)
2. **Job Success Rate** - % of successful jobs (green >99%, yellow >95%, red <95%)
3. **Queue Depth** - Current backlog in Redis queues
4. **Job Duration** - Latency percentiles (p50, p95, p99) per job type
5. **DTW Performance** - CPU vs GPU comparison
6. **GPU Memory Usage** - Allocated, reserved, and total memory per GPU
7. **Embedding Throughput** - V-JEPA2 vs heuristic embedder speed
8. **Active Workers** - Number of running worker pods
9. **Total Clips** - Cumulative clips indexed
10. **Failure Analysis** - Recent job failures by type

### Screenshot

*(Placeholder for dashboard screenshot)*

### Import Instructions

#### Via Kubernetes ConfigMap

```bash
# Create ConfigMap
kubectl create configmap sopilot-grafana-dashboard \
  --from-file=grafana-dashboard.json \
  -n monitoring

# Configure Grafana to load from ConfigMap
# (Add to Grafana deployment or Helm values)
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  sopilot.json: |-
    {{ .Files.Get "grafana-dashboard.json" | indent 4 }}
```

#### Via Grafana UI

1. Open Grafana web interface
2. Navigate to **Dashboards** > **Import**
3. Click **Upload JSON file**
4. Select `monitoring/grafana-dashboard.json`
5. Select Prometheus datasource
6. Click **Import**

## Prometheus Alerting Rules

### Alert Groups

#### 1. sopilot.jobs
- **HighJobFailureRate** - >10% job failure rate for 5 minutes (warning)
- **QueueBacklog** - Queue depth >100 for 5 minutes (warning)
- **CriticalQueueBacklog** - Queue depth >500 for 2 minutes (critical)
- **SlowJobProcessing** - p95 ingest duration >60s for 10 minutes (warning)

#### 2. sopilot.gpu
- **HighGPUMemoryUsage** - GPU memory >90% for 5 minutes (warning)
- **GPUNotAvailable** - GPU expected but not detected (critical)

#### 3. sopilot.performance
- **DTWPerformanceDegradation** - GPU DTW p95 >0.5s for 10 minutes (warning)
- **SlowEmbeddingGeneration** - V-JEPA2 p95 >10s for 10 minutes (warning)

#### 4. sopilot.system
- **NoActiveWorkers** - Zero workers active for 2 minutes (critical)
- **RedisConnectionErrors** - Redis errors >0.1/sec for 2 minutes (critical)
- **DatabaseLocked** - SQLite lock contention >1/sec for 5 minutes (warning)

#### 5. sopilot.sla
- **APIDown** - API health check failing for 1 minute (critical)
- **HighAPILatency** - API p95 >5s for 5 minutes (warning)

### Deployment

```bash
# Apply to Kubernetes (requires Prometheus Operator)
kubectl apply -f monitoring/prometheus-rules.yaml

# Verify rules are loaded
kubectl get prometheusrule -n sopilot

# Check Prometheus UI
# Navigate to: Status > Rules
# Look for group: sopilot.*
```

### Testing Alerts

```bash
# Trigger HighJobFailureRate alert
# (Inject failing jobs for testing)

# Check firing alerts in Prometheus UI
# Navigate to: Alerts

# Or query via PromQL
ALERTS{alertname="HighJobFailureRate",alertstate="firing"}
```

## Alertmanager Configuration

### Slack Integration

```yaml
# alertmanager.yml
receivers:
- name: 'slack'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#sopilot-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      *Severity:* {{ .Labels.severity }}
      *Summary:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      {{ end }}
```

### PagerDuty Integration

```yaml
receivers:
- name: 'pagerduty'
  pagerduty_configs:
  - service_key: YOUR_PAGERDUTY_INTEGRATION_KEY
    description: '{{ .GroupLabels.alertname }}'
```

### Email Integration

```yaml
receivers:
- name: 'email'
  email_configs:
  - to: 'ops@example.com'
    from: 'alertmanager@example.com'
    smarthost: 'smtp.gmail.com:587'
    auth_username: 'alertmanager@example.com'
    auth_password: 'YOUR_APP_PASSWORD'
    headers:
      Subject: 'SOPilot Alert: {{ .GroupLabels.alertname }}'
```

## Metrics Reference

### Counters
- `sopilot_ingest_jobs_total` - Total ingest jobs by status
- `sopilot_score_jobs_total` - Total score jobs by status
- `sopilot_training_jobs_total` - Total training jobs by trigger
- `sopilot_redis_connection_errors_total` - Redis connection failures
- `sopilot_database_locked_errors_total` - SQLite lock errors

### Histograms
- `sopilot_job_duration_seconds` - Job execution time (buckets: 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300)
- `sopilot_dtw_execution_seconds` - DTW alignment time (labels: gpu=true/false)
- `sopilot_embedding_generation_seconds` - Embedding generation time (labels: embedder=vjepa2/heuristic-v1)

### Gauges
- `sopilot_queue_depth` - Current jobs in queue (labels: queue=ingest/score/training)
- `sopilot_gpu_memory_bytes` - GPU memory usage (labels: device_id, memory_type=allocated/reserved/total)
- `sopilot_active_workers` - Number of active workers (labels: queue)
- `sopilot_total_clips` - Total indexed clips

### Info Metrics
- `sopilot_build_info` - Build metadata (labels: version, commit, build_date)

## PromQL Query Examples

### Job Throughput
```promql
# Ingest jobs per second (last 5 minutes)
rate(sopilot_ingest_jobs_total{status="completed"}[5m])

# Total jobs processed (last hour)
increase(sopilot_score_jobs_total{status="completed"}[1h])
```

### Performance
```promql
# 95th percentile DTW execution time (GPU)
histogram_quantile(0.95, rate(sopilot_dtw_execution_seconds_bucket{gpu="true"}[5m]))

# Average job duration
rate(sopilot_job_duration_seconds_sum{job_type="ingest"}[5m])
/ rate(sopilot_job_duration_seconds_count{job_type="ingest"}[5m])
```

### GPU Metrics
```promql
# GPU memory utilization %
(sopilot_gpu_memory_bytes{memory_type="allocated"} / sopilot_gpu_memory_bytes{memory_type="total"}) * 100

# GPU memory available (GB)
(sopilot_gpu_memory_bytes{memory_type="total"} - sopilot_gpu_memory_bytes{memory_type="allocated"}) / 1024 / 1024 / 1024
```

### Queue Analysis
```promql
# Queue depth by queue
sopilot_queue_depth

# Total pending jobs across all queues
sum(sopilot_queue_depth)

# Queue processing rate
rate(sopilot_queue_depth[5m])
```

## Troubleshooting

### Dashboard Not Loading

1. **Check Prometheus datasource:**
   ```bash
   # Test Prometheus connection
   curl http://prometheus-server/api/v1/query?query=up
   ```

2. **Verify metrics endpoint:**
   ```bash
   # Check SOPilot API metrics
   kubectl port-forward svc/sopilot-api 8000:80 -n sopilot
   curl http://localhost:8000/metrics
   ```

3. **Check Grafana logs:**
   ```bash
   kubectl logs -f deployment/grafana -n monitoring
   ```

### Alerts Not Firing

1. **Check PrometheusRule is loaded:**
   ```bash
   kubectl get prometheusrule sopilot-alerts -n sopilot -o yaml
   ```

2. **Verify rules in Prometheus UI:**
   - Navigate to **Status > Rules**
   - Look for `sopilot.*` groups

3. **Check Alertmanager config:**
   ```bash
   kubectl get configmap alertmanager-config -n monitoring -o yaml
   ```

### Missing Metrics

1. **Check ServiceMonitor:**
   ```bash
   kubectl get servicemonitor -n sopilot
   ```

2. **Verify Prometheus targets:**
   - Navigate to **Status > Targets**
   - Look for `sopilot-api` endpoint

3. **Check metrics collection:**
   ```bash
   # Query Prometheus for SOPilot metrics
   curl 'http://prometheus-server/api/v1/query?query=sopilot_ingest_jobs_total'
   ```

## Advanced Configuration

### Custom Dashboards

Create custom panels by editing `grafana-dashboard.json`:

```json
{
  "id": 11,
  "title": "Custom Metric",
  "type": "graph",
  "gridPos": {"x": 0, "y": 32, "w": 12, "h": 8},
  "targets": [
    {
      "expr": "your_custom_promql_query",
      "legendFormat": "{{label}}",
      "refId": "A"
    }
  ]
}
```

### Alert Tuning

Adjust thresholds in `prometheus-rules.yaml`:

```yaml
# Reduce sensitivity (higher threshold)
- alert: HighJobFailureRate
  expr: (sum(rate(...)) / sum(rate(...))) > 0.20  # Changed from 0.10

# Increase evaluation time
  for: 10m  # Changed from 5m
```

### Multi-Cluster Monitoring

For federated Prometheus across clusters:

```yaml
# prometheus-federation.yaml
scrape_configs:
- job_name: 'federate'
  scrape_interval: 15s
  honor_labels: true
  metrics_path: '/federate'
  params:
    'match[]':
      - '{job="sopilot-api"}'
      - '{__name__=~"sopilot_.*"}'
  static_configs:
  - targets:
    - 'cluster1-prometheus:9090'
    - 'cluster2-prometheus:9090'
```

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Operator](https://prometheus-operator.dev/)
- [Alertmanager Configuration](https://prometheus.io/docs/alerting/latest/configuration/)
