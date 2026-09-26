#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --partition=hpc7g
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=smoke-test-%j.out
#
# Two-node check that MPI runs over EFA and bin/summa gives correct results across nodes.
# Run after infra/install-software.sbatch:   sbatch infra/smoke-test.sbatch
set -euo pipefail

OMPI_PREFIX=${OMPI_PREFIX:-/opt/amazon/openmpi}
export PATH=$OMPI_PREFIX/bin:/opt/amazon/efa/bin:$PATH
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PEAK_NODE=2662   # GFLOP/s: 64 cores x 2.6 GHz x 16 DP FLOP/cycle (4x128-bit FMA pipes)
CORES=$(nproc)

echo "== nodes: $SLURM_JOB_NODELIST, $CORES cores each"
echo "== EFA provider:"
fi_info -p efa -t FI_EP_RDM | head -4

# Force Open MPI onto libfabric/EFA: if EFA is unusable the job fails instead of
# silently falling back to TCP.
export FI_PROVIDER=efa
MPIRUN=(mpirun --mca pml cm --mca mtl ofi --bind-to core)

echo "== 1 rank x $CORES threads per node"
"${MPIRUN[@]}" -np 2 --map-by "ppr:1:node:PE=$CORES" \
  ./bin/summa -n 8192 -t "$CORES" --verify --csv -p "$PEAK_NODE"

echo "== 2 ranks x $((CORES / 2)) threads per node"
"${MPIRUN[@]}" -np 4 --map-by "ppr:2:node:PE=$((CORES / 2))" \
  ./bin/summa -n 8192 -t $((CORES / 2)) --verify --csv -p "$PEAK_NODE"

echo "== same, blocking broadcasts"
"${MPIRUN[@]}" -np 4 --map-by "ppr:2:node:PE=$((CORES / 2))" \
  ./bin/summa -n 8192 -t $((CORES / 2)) --verify --no-overlap
