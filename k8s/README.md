# SOPilot Kubernetes Deployment

## Quick Start

### Prerequisites

- Kubernetes cluster 1.25+ with GPU support (NVIDIA GPU Operator)
- `kubectl` configured to connect to your cluster
- Storage provisioner (for PersistentVolumeClaims)
- Ingress controller (nginx, traefik, etc.)

### Deploy with Kustomize

```bash
# Build and preview manifests
kubectl kustomize k8s/

# Apply all resources
kubectl apply -k k8s/

# Check deployment status
kubectl get all -n sopilot

# Watch pods come up
kubectl get pods -n sopilot -w
```

### Deploy Manually

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (IMPORTANT: Update secret.yaml first!)
kubectl apply -f k8s/secret.yaml

# Deploy infrastructure
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/statefulset-redis.yaml

# Wait for Redis to be ready
kubectl wait --for=condition=ready pod -l component=redis -n sopilot --timeout=120s

# Deploy application
kubectl apply -f k8s/deployment-api.yaml
kubectl apply -f k8s/deployment-worker-gpu.yaml
kubectl apply -f k8s/deployment-worker-cpu.yaml

# Expose externally
kubectl apply -f k8s/ingress.yaml
```

## Configuration

### 1. Update Secrets

**CRITICAL:** Replace placeholder tokens in `secret.yaml` before deploying:

```bash
# Generate secure tokens
kubectl create secret generic sopilot-secrets \
  --from-literal=SOPILOT_API_ROLE_TOKENS="admin:$(openssl rand -hex 32),operator:$(openssl rand -hex 32),viewer:$(openssl rand -hex 32)" \
  --from-literal=SOPILOT_AUDIT_SIGNING_KEY="$(openssl rand -hex 32)" \
  --from-literal=SOPILOT_AUDIT_SIGNING_KEY_ID="prod" \
  -n sopilot --dry-run=client -o yaml > k8s/secret.yaml
```

### 2. Update Ingress Hostname

Edit `k8s/ingress.yaml`:

```yaml
spec:
  rules:
  - host: sopilot.your-domain.com  # Replace with your domain
```

### 3. Configure Storage

Edit `k8s/pvc.yaml` to match your storage class:

```yaml
spec:
  storageClassName: fast-ssd  # Change to your storage class
  resources:
    requests:
      storage: 500Gi  # Adjust based on video volume
```

Check available storage classes:

```bash
kubectl get storageclass
```

### 4. Scale Workers

Adjust replicas in deployment manifests:

```yaml
# deployment-worker-gpu.yaml
spec:
  replicas: 2  # Increase for more GPU workers

# deployment-worker-cpu.yaml
spec:
  replicas: 4  # Increase for more CPU workers
```

Or scale dynamically:

```bash
kubectl scale deployment sopilot-worker-gpu --replicas=4 -n sopilot
kubectl scale deployment sopilot-worker-cpu --replicas=8 -n sopilot
```

## GPU Node Setup

### Enable NVIDIA GPU Support

```bash
# Install NVIDIA GPU Operator (for supported K8s distributions)
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# Verify GPU nodes
kubectl get nodes -l nvidia.com/gpu=true

# Check GPU availability
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu
```

### Label GPU Nodes (Optional)

```bash
# Label nodes by GPU type
kubectl label node <node-name> gpu-type=rtx-5090

# Verify
kubectl get nodes -L gpu-type
```

## Verification

### Check Pod Status

```bash
# All pods should be Running
kubectl get pods -n sopilot

# Check logs
kubectl logs -f deployment/sopilot-api -n sopilot
kubectl logs -f deployment/sopilot-worker-gpu -n sopilot
kubectl logs -f deployment/sopilot-worker-cpu -n sopilot
```

### Test API

```bash
# Port forward to API
kubectl port-forward svc/sopilot-api 8000:80 -n sopilot

# Test health endpoint
curl http://localhost:8000/health

# Test metrics
curl http://localhost:8000/metrics
```

### Check GPU Allocation

```bash
# Verify GPU is allocated to worker pods
kubectl describe pod -l worker-type=gpu -n sopilot | grep nvidia.com/gpu

# Check CUDA visibility inside pod
kubectl exec -it deployment/sopilot-worker-gpu -n sopilot -- nvidia-smi
```

## Monitoring

### Prometheus Metrics

SOPilot exposes metrics at `/metrics` endpoint. To scrape with Prometheus:

```yaml
# prometheus-servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sopilot
  namespace: sopilot
spec:
  selector:
    matchLabels:
      app: sopilot
      component: api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

Apply:

```bash
kubectl apply -f prometheus-servicemonitor.yaml
```

### Grafana Dashboard

Import the provided dashboard:

```bash
kubectl create configmap sopilot-grafana-dashboard \
  --from-file=monitoring/grafana-dashboard.json \
  -n monitoring
```

## Troubleshooting

### Pods Stuck in Pending

```bash
# Check events
kubectl describe pod <pod-name> -n sopilot

# Common causes:
# - Insufficient resources (CPU, memory, GPU)
# - PVC not bound (check storage provisioner)
# - Image pull errors (check image name)
```

### GPU Workers Not Starting

```bash
# Verify GPU nodes exist
kubectl get nodes -l nvidia.com/gpu=true

# Check GPU availability
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Ensure NVIDIA GPU Operator is running
kubectl get pods -n gpu-operator
```

### Redis Connection Errors

```bash
# Check Redis pod
kubectl get pods -l component=redis -n sopilot

# Test connectivity from worker
kubectl exec -it deployment/sopilot-worker-cpu -n sopilot -- \
  python -c "import redis; r=redis.from_url('redis://sopilot-redis:6379/0'); print(r.ping())"
```

### High Memory Usage

```bash
# Check resource usage
kubectl top pods -n sopilot

# Adjust resource limits in deployment manifests
# Or enable horizontal pod autoscaling (HPA)
```

## Scaling

### Horizontal Pod Autoscaler (HPA)

```bash
# Auto-scale API pods based on CPU
kubectl autoscale deployment sopilot-api \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n sopilot

# Check HPA status
kubectl get hpa -n sopilot
```

### Vertical Pod Autoscaler (VPA)

For automatic resource adjustment, install VPA:

```bash
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Create VPA for API
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: sopilot-api-vpa
  namespace: sopilot
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sopilot-api
  updatePolicy:
    updateMode: "Auto"
EOF
```

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/

# Or delete namespace (removes everything)
kubectl delete namespace sopilot
```

## Production Checklist

- [ ] Secrets updated with strong random tokens
- [ ] Ingress hostname configured
- [ ] TLS/SSL certificates configured (cert-manager)
- [ ] Storage class configured (SSD recommended)
- [ ] Resource limits tuned for workload
- [ ] GPU nodes labeled and schedulable
- [ ] Prometheus monitoring enabled
- [ ] Grafana dashboards imported
- [ ] Backup strategy for PVCs
- [ ] Network policies defined (if required)
- [ ] RBAC roles configured (if needed)
- [ ] Log aggregation configured (ELK, Loki, etc.)
