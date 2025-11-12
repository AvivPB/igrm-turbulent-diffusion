#!/bin/bash -l
#########################################################
#SBATCH -J identify_halo_progenitors

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
#SBATCH -o /scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_snap/slurm_files/slurm-%j.out
#########################################################
#SBATCH --time=2:00:00
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


source /scratch/aspadawe/igrm-turbulent-diffusion/pyenvs/main/bin/activate


snap_dir=/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/
snap_base=snapshot_
caesar_dir=/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_snap/
caesar_base=caesar_
caesar_suffix=''
source_snap_num=151
# target_snap_nums=$(seq 0 1 151)
source_halo_ids=0
n_most=2
nproc=32
output_file=/scratch/aspadawe/igrm-turbulent-diffusion/snapshots/HyenasC/L1/SimbaC_L1_Calibration/halo_3224_grad/caesar_snap/halo_progen_info


echo target_snap_nums:
echo $target_snap_nums


echo
echo 'Identifying Progenitors'
echo

python identify_halo_progenitors.py --snap_dir=$snap_dir --snap_base=$snap_base --caesar_dir=$caesar_dir --caesar_base=$caesar_base --caesar_suffix=$caesar_suffix --source_snap_nums=$source_snap_nums --target_snap_nums {151..0} --source_halo_ids $source_halo_ids --n_most=$n_most --nproc=$nproc --output_file=$output_file --clear_output_file

echo
echo 'done'

########################################################
