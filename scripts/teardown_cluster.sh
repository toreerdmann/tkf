#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-tkf-dev}"
REGISTRY_NAME="${REGISTRY_NAME:-tkf-registry}"

echo "=== Tearing down k3d cluster '${CLUSTER_NAME}' ==="

if k3d cluster list "${CLUSTER_NAME}" &> /dev/null; then
    k3d cluster delete "${CLUSTER_NAME}"
    echo "Cluster '${CLUSTER_NAME}' deleted."
else
    echo "Cluster '${CLUSTER_NAME}' does not exist."
fi

# Optional registry teardown if passed --with-registry
if [[ "${1:-}" == "--with-registry" ]]; then
    if k3d registry list "${REGISTRY_NAME}" &> /dev/null; then
        k3d registry delete "${REGISTRY_NAME}"
        echo "Registry '${REGISTRY_NAME}' deleted."
    fi
fi
