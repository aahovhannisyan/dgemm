# AWS cluster for the distributed DGEMM benchmarks

A Slurm cluster from AWS ParallelCluster 3:

- **Head node:** `c7g.xlarge`.
- **Compute nodes:** up to `MAX_NODES` × `hpc7g.16xlarge`, each with 64 Graviton3E cores, 128 GiB and 200 Gbps EFA. They sit in one placement group and scale to zero when idle.
- **Shared storage:** a `/shared` EBS volume that holds Spack, the baselines, a Python venv, this repo and the results.

| File | Purpose |
|---|---|
| `pcluster.yaml.template` | Cluster definition with `${...}` placeholders |
| `cluster.env.example` | Your account-specific values; copy it to `cluster.env` (git-ignored) |
| `render-config.sh` | Writes `pcluster.yaml` (git-ignored) from the template and `cluster.env` |
| `spack.yaml` | Spack environment: OpenBLAS, ArmPL, ScaLAPACK, COSMA, SLATE, Python + numpy + mpi4py |
| `install-software.sbatch` | One-time install on a compute node, which also builds this repo |
| `smoke-test.sbatch` | Two-node check of EFA and `bin/summa --verify` |

## 1. One-time prerequisites (on your machine)

1. **AWS CLI credentials** for the target account:
   ```bash
   aws sts get-caller-identity
   ```
2. **The ParallelCluster CLI.** It needs Node.js for the CDK:
   ```bash
   brew install node
   ```
   ```bash
   python3 -m venv ~/.venvs/pcluster && ~/.venvs/pcluster/bin/pip install "aws-parallelcluster==3.16.*"
   ```
   ```bash
   export PATH=~/.venvs/pcluster/bin:$PATH
   ```
3. **The HPC instance quota.** HPC instance types have their own quota, *Running On-Demand HPC instances*, counted in vCPUs. Each hpc7g node is 64 vCPUs, so 16 nodes need 1024. Request it in the Service Quotas console for your region. Approval can take a day.
4. **An availability zone that offers hpc7g.** The HPC families are offered in only some AZs, and EFA does not work across AZs:
   ```bash
   aws ec2 describe-instance-type-offerings --region us-east-1 --location-type availability-zone --filters Name=instance-type,Values=hpc7g.16xlarge --query 'InstanceTypeOfferings[].Location' --output text
   ```
5. **A key pair:**
   ```bash
   aws ec2 create-key-pair --region us-east-1 --key-name dgemm-key --query KeyMaterial --output text > ~/.ssh/dgemm-key.pem && chmod 600 ~/.ssh/dgemm-key.pem
   ```
6. **Networking.** You need a public subnet for the head node and a private subnet with a NAT gateway for the compute nodes, both in the AZ from step 4. The easiest way is the configure wizard's "automate VPC creation" option. Pick the AZ from step 4 and "Head node in a public subnet and compute fleet in a private subnet". Throw away the config it writes, but note the two subnet IDs:
   ```bash
   pcluster configure --config /tmp/pcluster-wizard.yaml
   ```

## 2. Create the cluster

```bash
cp infra/cluster.env.example infra/cluster.env
```
Edit `infra/cluster.env` to fill in the region, the subnets, the key name, your IP and `MAX_NODES`. Then:
```bash
infra/render-config.sh
```
```bash
pcluster create-cluster -n dgemm -c infra/pcluster.yaml --dryrun true
```
```bash
pcluster create-cluster -n dgemm -c infra/pcluster.yaml
```
Wait for `CREATE_COMPLETE`, which takes about 15–20 minutes:
```bash
pcluster describe-cluster -n dgemm --query clusterStatus
```

## 3. Install the software (on the cluster)

Copy this repo to the shared volume. rsync works for a private repo and includes uncommitted changes:
```bash
HEAD_IP=$(pcluster describe-cluster -n dgemm --query headNode.publicIpAddress | tr -d '"')
```
```bash
rsync -av --exclude bin --exclude .venv --exclude .idea -e "ssh -i ~/.ssh/dgemm-key.pem" ./ ec2-user@$HEAD_IP:/shared/dgemm/
```
Then log in:
```bash
pcluster ssh -n dgemm -i ~/.ssh/dgemm-key.pem
```
On the head node, submit the install job:
```bash
cd /shared/dgemm && sbatch infra/install-software.sbatch
```
The job starts a compute node, which takes a few minutes. It then:
- clones Spack v1.2.2
- builds the environment in `infra/spack.yaml` against the AMI's EFA-enabled Open MPI (`/opt/amazon/openmpi`)
- creates `/shared/venv-dgemm` with Dask
- builds and tests this repo

Expect it to run for an hour or more; most of that is building Python, SLATE and COSMA. Watch it with:
```bash
tail -f install-software-*.out
```
Re-running the job is safe: packages that are already installed are reused.

To use the libraries from a shell or job script:
```bash
. /shared/spack/share/spack/setup-env.sh && spack env activate /shared/spack-envs/dgemm
```
For the Python baselines, use `/shared/venv-dgemm/bin/python`.

## 4. Smoke test

```bash
sbatch infra/smoke-test.sbatch
```

The output should show:
- the `efa` libfabric provider
- `Verify: ... (OK)` for 1 rank × 64 threads per node and for 2 ranks × 32 threads per node
- a CSV line with `pct_peak`

The job sets `FI_PROVIDER=efa` and `--mca mtl ofi`, so a broken EFA setup fails the job instead of silently falling back to TCP.

## Cost

These are approximate us-east-1 on-demand prices; check the pricing pages for your region.

| Resource | While it exists |
|---|---|
| hpc7g.16xlarge compute node | about $1.68/h each, **only while a job holds it** (plus 10 min idle) |
| c7g.xlarge head node | about $0.15/h |
| NAT gateway (from the wizard) | about $0.045/h plus data processed |
| 200 GiB gp3 `/shared` | about $16/month |

A 16-node strong-scaling sweep that takes 2 hours of wall time costs about 16 × 2 × $1.68 ≈ $54.

Between sessions you can stop the compute fleet:
```bash
pcluster update-compute-fleet -n dgemm --status STOP_REQUESTED
```
The head node, NAT gateway and EBS volume keep billing until you delete them.

## Teardown

1. Copy the results off. `/shared` is deleted with the cluster:
   ```bash
   rsync -av -e "ssh -i ~/.ssh/dgemm-key.pem" ec2-user@$HEAD_IP:/shared/dgemm/results/ ./results/
   ```
2. Delete the cluster:
   ```bash
   pcluster delete-cluster -n dgemm
   ```
3. If the wizard created the VPC, delete its `parallelclusternetworking-*` CloudFormation stack too. It holds the NAT gateway.
4. Check that nothing is left behind:
   ```bash
   aws ec2 describe-instances --region us-east-1 --filters Name=tag:project,Values=dgemm-bench Name=instance-state-name,Values=running,stopped --query 'Reservations[].Instances[].InstanceId'
   ```
   ```bash
   aws ec2 describe-volumes --region us-east-1 --filters Name=tag:project,Values=dgemm-bench --query 'Volumes[].VolumeId'
   ```

## Troubleshooting

- **`missing on the compute node image: gfortran`** (or another tool). Compute nodes are recreated on every scale-up, so a manual `dnf install` doesn't stick. Add an `OnNodeConfigured` custom action to the `hpc7g` queue that runs `dnf install -y gcc-gfortran`. The script must be hosted on S3 or HTTPS. Alternatively, build a custom AMI with `pcluster build-image`.
- **`InsufficientInstanceCapacity`** or nodes stuck in `CF`/`down~` state. hpc7g capacity per AZ is limited. Retry later or lower `MAX_NODES`. `sinfo` and `/var/log/parallelcluster/slurm_resume.log` on the head node show the cause.
- **Quota errors on scale-up.** Check the *Running On-Demand HPC instances* quota from step 1.3.
