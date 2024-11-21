#!/bin/bash
#SBATCH --job-name=EEG_Process_Bundled
#SBATCH --output=logs/output_%A_%a.txt
#SBATCH --error=logs/error_%A_%a.txt
#SBATCH --ntasks=1                    # One task per job
#SBATCH --cpus-per-task=1              # Number of CPU cores per task
#SBATCH --mem=50G                      # Memory per task
#SBATCH --time=10:00:00                # Time limit for each job
#SBATCH --array=1-30                   # Parametric job array (you can change this to 30 or any number)
#SBATCH --partition=single

# Load necessary modules
module load devel/miniconda/23.9.0-py3.9.15  # Adjust based on your environment

# Activate the conda environment
source activate matin

# Define subject and session ranges
SUBJECT_START=326
SUBJECT_END=440
TOTAL_SUBJECTS=$((SUBJECT_END - SUBJECT_START + 1))
SESSIONS_PER_SUBJECT=25

# Parametric value for the number of array jobs
ARRAY_JOBS=30  # Adjust this value as needed (number of jobs in the array)

# Calculate the number of subjects to process per job
SUBJECTS_PER_JOB=$((TOTAL_SUBJECTS / ARRAY_JOBS))
REMAINING_SUBJECTS=$((TOTAL_SUBJECTS % ARRAY_JOBS))

# Calculate the start and end subjects for this job
task_id=$SLURM_ARRAY_TASK_ID
start_subject_id=$((SUBJECT_START + (task_id - 1) * SUBJECTS_PER_JOB))
end_subject_id=$((start_subject_id + SUBJECTS_PER_JOB - 1))

# Handle the remaining subjects (if any) in the last jobs
if [ $task_id -le $REMAINING_SUBJECTS ]; then
    # Distribute remaining subjects to the first few jobs
    start_subject_id=$((start_subject_id + (task_id - 1)))
    end_subject_id=$((end_subject_id + task_id))
else
    # Adjust for jobs beyond the remaining subjects
    start_subject_id=$((start_subject_id + REMAINING_SUBJECTS))
    end_subject_id=$((end_subject_id + REMAINING_SUBJECTS))
fi

echo "Processing subjects from $start_subject_id to $end_subject_id"

# Loop through each subject and session, running the Python script
for subject_id in $(seq $start_subject_id $end_subject_id); do
    for session_id in $(seq 1 $SESSIONS_PER_SUBJECT); do
        echo "Processing subject $subject_id, session $session_id"
        python data_collection.py $subject_id $session_id
    done
done
