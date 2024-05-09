# Project Location
```bash
cd /nfs/gce/projects/M2dt-OED/
```

# Read only Firedrake virtual environment
```bash
. /nfs/gce/software/custom/linux-ubuntu22.04-x86_64/firedrake/firedrake/bin/activate
```

# Firedrake environment with additional packages
```bash
source /nfs/gce/projects/M2dt-OED/activate
```

I am still working on how best to manage a hybrid environment where we can add additional packages if needed but retain the read only base environment as well. The current implementation uses the libraries from another environment copied into `M2dt-OED/my_packages`.

# Start remote jupyter notebook
## On remote machine
```bash
jupyter notebook --no-browser --port=XXXX
```
Change port or do not specify for default port (`8888`).

## On local machine
```bash
ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost
```
Specify local port `YYYY` and previously selected remote port `XXXX`.

# Fenics environment
```bash
module load anaconda3
conda activate /nfs/gce/projects/M2dt-OED/fenics-env
```
