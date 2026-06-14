#!/bin/bash
# Build and push all container images to the local registry.
# Run from the repo root on the GPU server.
#
# Environment variables:
#   REGISTRY      — local container registry (default: localhost:5050)
#   EDGE_REGISTRY — registry address reachable from the edge/motor node
#                   (no default — set this to <gpu-server-ip>:5050 or similar)

set -euo pipefail

REGISTRY="${REGISTRY:-localhost:5050}"
: "${EDGE_REGISTRY:?Set EDGE_REGISTRY to the registry address reachable from the edge node (e.g. <gpu-server-ip>:5050)}"

echo "=== Building inference image ==="
docker build -f deploy/robot/docker/Dockerfile.inference -t so101-inference:latest .
docker tag so101-inference:latest ${REGISTRY}/so101-inference:latest
docker push ${REGISTRY}/so101-inference:latest

echo "=== Building camera node image ==="
docker build -f deploy/robot/docker/Dockerfile.camera -t so101-camera:latest .
docker tag so101-camera:latest ${REGISTRY}/so101-camera:latest
docker push ${REGISTRY}/so101-camera:latest

echo "=== Building MCP server image ==="
docker build -f deploy/robot/docker/Dockerfile.mcp -t so101-mcp-server:latest .
docker tag so101-mcp-server:latest ${REGISTRY}/so101-mcp-server:latest
docker push ${REGISTRY}/so101-mcp-server:latest

echo "=== Building motor node image ==="
docker build -f deploy/robot/docker/Dockerfile.motor -t so101-motor:latest .
docker tag so101-motor:latest ${EDGE_REGISTRY}/so101-motor:latest
docker push ${EDGE_REGISTRY}/so101-motor:latest

echo "=== All images built and pushed ==="
echo ""
echo "To deploy:"
echo "  kubectl apply -f deploy/robot/k8s/robot-brain.yaml"
echo "  kubectl get pods -n robot -w"
