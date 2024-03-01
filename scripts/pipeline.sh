#!/bin/bash

set -e

# Simulate 3 conditions
echo -e "\n> Simulating coherent condition\n"
python3 scripts/simulate.py --outdir results --condition coherent --dts 50
echo -e "\n> Simulating incoherent condition\n"
python3 scripts/simulate.py --outdir results --condition incoherent --dts 50
echo -e "\n> Simulating non-adapting condition\n"
python3 scripts/simulate.py --outdir results --condition nonadap --dts 50

# Generate plots
echo -e "\n> Plotting results\n"
python3 scripts/plot_activity.py --outdir results --indir results

# Calculate and plot BOLD
echo -e "\n> Calculating BOLD signal\n"
python3 scripts/bold.py --outdir results --indir results --dts 50

echo -e "\nAll done!\n"