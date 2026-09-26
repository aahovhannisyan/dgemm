#!/usr/bin/env bash
# Render infra/pcluster.yaml from pcluster.yaml.template and the values in infra/cluster.env.
set -euo pipefail
cd "$(dirname "$0")"

ENV_FILE=${1:-cluster.env}
[[ -f $ENV_FILE ]] || { echo "copy cluster.env.example to $ENV_FILE and fill it in" >&2; exit 1; }
set -a
. "./$ENV_FILE"
set +a

VARS=(AWS_REGION HEAD_SUBNET_ID COMPUTE_SUBNET_ID SSH_KEY_NAME ALLOWED_CIDR MAX_NODES)
for v in "${VARS[@]}"; do
  [[ -n ${!v:-} ]] || { echo "$v is not set in $ENV_FILE" >&2; exit 1; }
done

# only substitute our variables, never anything else that looks like $FOO
envsubst "$(printf '${%s} ' "${VARS[@]}")" < pcluster.yaml.template > pcluster.yaml
echo "wrote infra/pcluster.yaml"
