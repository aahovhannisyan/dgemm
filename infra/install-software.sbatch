#!/bin/bash
#SBATCH --job-name=install-software
#SBATCH --partition=hpc7g
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=install-software-%j.out
#
# One-time software setup, run on a compute node so everything is built for Graviton3E.
# Installs Spack and the baseline libraries (infra/spack.yaml), a Python venv with Dask,
# and builds this repo -- all under /shared, which every node mounts.
# From the repo checkout on the head node:   sbatch infra/install-software.sbatch
# Safe to re-run: finished Spack installs are reused.
set -euo pipefail

SHARED=${SHARED:-/shared}
SPACK_VERSION=${SPACK_VERSION:-v1.2.2}
OMPI_PREFIX=${OMPI_PREFIX:-/opt/amazon/openmpi}
REPO_DIR=${SLURM_SUBMIT_DIR:-$PWD}
ENV_DIR=$SHARED/spack-envs/dgemm
VENV=$SHARED/venv-dgemm
JOBS=$(nproc)

echo "== $(hostname), $(nproc) cores, repo $REPO_DIR"

# ---- prerequisites on the AMI -------------------------------------------------------------
missing=()
for tool in gcc g++ gfortran git make python3; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
if ((${#missing[@]})); then
  echo "missing on the compute node image: ${missing[*]}" >&2
  echo "Compute nodes are recreated on every scale-up, so install these through a custom AMI" >&2
  echo "or an OnNodeConfigured custom action rather than by hand (see infra/README.md)." >&2
  exit 1
fi
[[ -x $OMPI_PREFIX/bin/mpicc ]] || { echo "no Open MPI at $OMPI_PREFIX" >&2; exit 1; }
export PATH=$OMPI_PREFIX/bin:$PATH
OMPI_VERSION=$(ompi_info --parsable | awk -F: '$1=="ompi" && $2=="version" && $3=="full" {print $4; exit}')
echo "== Open MPI $OMPI_VERSION at $OMPI_PREFIX"

# ---- Spack + baseline libraries --------------------------------------------------------------
if [[ ! -d $SHARED/spack ]]; then
  git clone --depth=2 --branch "$SPACK_VERSION" https://github.com/spack/spack.git "$SHARED/spack"
fi
. "$SHARED/spack/share/spack/setup-env.sh"

# recreate the environment from the repo's spack.yaml each run; installed packages are kept
rm -rf "$ENV_DIR"
mkdir -p "$(dirname "$ENV_DIR")"
spack env create -d "$ENV_DIR" "$REPO_DIR/infra/spack.yaml"
spack env activate "$ENV_DIR"
spack compiler find
cat > "$ENV_DIR/openmpi-external.yaml" <<EOF
packages:
  openmpi:
    externals:
      - spec: openmpi@$OMPI_VERSION fabrics=ofi schedulers=slurm
        prefix: $OMPI_PREFIX
EOF
spack config add -f "$ENV_DIR/openmpi-external.yaml"
spack concretize --force
spack install --fail-fast -j "$JOBS"
spack env deactivate

# ---- Python venv for Dask, on top of Spack's python/numpy/mpi4py -------------------------------
PYVIEW=$SHARED/spack-envs/dgemm-python-view
if [[ ! -x $VENV/bin/python ]]; then
  "$PYVIEW/bin/python3" -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install --upgrade "dask[distributed]" dask-mpi
"$VENV/bin/python" -c 'import numpy, mpi4py, dask; print("numpy", numpy.__version__, "| mpi4py", mpi4py.__version__, "| dask", dask.__version__)'

# ---- this repo -------------------------------------------------------------------------------
cd "$REPO_DIR"
make clean
make
make summa MPICC="$OMPI_PREFIX/bin/mpicc"
make test

echo "== done"
