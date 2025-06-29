#!/bin/bash

echo "Sync Start"
rsync -avz ./exec_scripts/ schrape@login.ae.iti.kit.edu:/nfs/home/schrape/slurm_output/



ssh schrape@login.ae.iti.kit.edu << 'EOF'
SERVER=$(sinfo --Node --partition=all --long --noheader | awk '$1 != "iverson"  && ($4 == "idle" || $4 == "idle~" || $4 == "idle#") {print $1, $5}' | sort -k2 -nr | head -n 1 | awk '{print $1}')
SCRIPT=train.sh


if [ -n "$SERVER" ]; then
    echo "Selected server: $SERVER with the most available CPUs."
    echo "Starting $SCRIPT job on $SERVER."
    cd slurm_output || exit
    sbatch --partition=$SERVER $SCRIPT
else
    echo "No suitable server is available."
fi
EOF
