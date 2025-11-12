#!/bin/bash -l
#########################################################
#SBATCH -J identify_halos-Hyenas_L1_halo_3224

#SBATCH --mail-user=apadawer@uvic.ca
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
##SBATCH -o /project/b/babul/aspadawe/snapshots/Hyenas/L1/halo_3224/caesar/slurm_files/slurm-%j.out
#########################################################
#SBATCH --time=1:00:00
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

module load NiaEnv/2022a gcc/11.3.0 openssl/1.1.1k sqlite/3.35.5 hdf5/1.12.3

main_dir=/home/b/babul/wcui/runs/HYENAS/
caesar_dir=Groups/

# DIRS=$(find $main_dir -type d)

for dir in $main_dir; do

        echo $dir
        echo

        # /scratch/b/babul/rennehan/for_aviv/python-3.13.5/bin/python identify_halo.py --caesar_dir=$dir$caesar_dir --snap_nums=151 --caesar_base=Caesar_halo_3224_ --output_file=/project/b/babul/aspadawe/snapshots/Hyenas/L1/halo_3224/caesar/halo_info --target_property=m500c --target_value_min=1e12 --target_value_max=1e15 --target_units=Msun --contamination_min=0 --contamination_max=0

        echo
        echo
done

########################################################
