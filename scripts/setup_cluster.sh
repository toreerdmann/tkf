#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-tkf-dev}"
REGISTRY_NAME="${REGISTRY_NAME:-tkf-registry}"
REGISTRY_PORT="${REGISTRY_PORT:-5111}"

echo "=== Setting up k3d cluster for tkf development ==="
echo "Cluster Name:   ${CLUSTER_NAME}"
echo "Registry Name:  ${REGISTRY_NAME}"
echo "Registry Port:  ${REGISTRY_PORT}"

# Check if k3d is installed
if ! command -v k3d &> /dev/null; then
    echo "Error: k3d is not installed. Please install k3d first."
    exit 1
fi

# Check if cluster already exists
if k3d cluster list "${CLUSTER_NAME}" &> /dev/null; then
    echo "Cluster '${CLUSTER_NAME}' already exists."
    echo "Switching kubectl context to k3d-${CLUSTER_NAME}..."
    kubectl config use-context "k3d-${CLUSTER_NAME}"
    exit 0
fi

# Create local registry if not exists
if ! k3d registry list "${REGISTRY_NAME}" &> /dev/null; then
    echo "Creating local registry '${REGISTRY_NAME}' on localhost:${REGISTRY_PORT}..."
    k3d registry create "${REGISTRY_NAME}" --port "${REGISTRY_PORT}"
else
    echo "Local registry '${REGISTRY_NAME}' already exists."
fi

# Create k3d cluster connected to the registry
echo "Creating k3d cluster '${CLUSTER_NAME}'..."
k3d cluster create "${CLUSTER_NAME}" \
    --registry-use "${REGISTRY_NAME}:${REGISTRY_PORT}" \
    --agents 1 \
    --port "8080:80@loadbalancer" \
    --wait

echo "Switching kubectl context to k3d-${CLUSTER_NAME}..."
kubectl config use-context "k3d-${CLUSTER_NAME}"

echo "Verifying cluster readiness..."
kubectl wait --for=condition=Ready nodes --all --timeout=60s

echo "=== Cluster '${CLUSTER_NAME}' is ready! ==="
echo "Local container registry: localhost:${REGISTRY_PORT}"
echo "In-cluster registry ref:  ${REGISTRY_NAME}:${REGISTRY_PORT}"
