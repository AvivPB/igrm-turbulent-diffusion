#!/bin/bash -l
#########################################################
#SBATCH -J make_caesar_files

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L0_Calibration/halo_3224_smag_og_good/caesar_snap/slurm_files/slurm-%j.out
#########################################################
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=0
#########################################################

##SBATCH --array=1-10%1 # Run a N-job array, 1 job at a time
#########################################################


# ---------------------------------------------------------------------
#echo "Current working directory: `pwd`"
#echo "Starting run at: `date`"
# ---------------------------------------------------------------------
#echo ""
#echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
#echo "Job task $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
#echo ""
# ---------------------------------------------------------------------

# module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate
echo
which python
echo

snap_dir=/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L0_Calibration/halo_3224_smag_og_good/
caesar_dir=/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L0_Calibration/halo_3224_smag_og_good/caesar_snap/

python make_caesar_files.py --snap_dir=$snap_dir --snap_nums {151..0} --snap_base=snapshot_ --caesar_dir=$caesar_dir --haloid=snap --blackholes --aperture=30 --half_stellar_radius_property --nproc=32 --lowres=2

########################################################
