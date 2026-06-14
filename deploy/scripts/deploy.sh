#!/bin/bash
# Deploy or redeploy all k3s resources.
# Run from repo root on the GPU server.

set -euo pipefail

echo "=== Deploying SO-101 demo ==="

# Apply namespace + deployments
kubectl apply -f deploy/robot/k8s/robot-brain.yaml

echo ""
echo "Watching pods..."
kubectl get pods -n robot -w
