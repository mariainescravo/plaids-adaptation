.PHONY: all
all: plots bold

.PHONY: sims
sims: results/coherent.csv results/incoherent.csv results/nonadap.csv

%.csv: scripts/simulate.py src/circuit.py src/stimulus.py
	python3 $< --outdir results --condition $(patsubst %.csv,%,$(notdir $@)) --dts 50

plots: scripts/plot_activity.py results/coherent.csv results/incoherent.csv results/nonadap.csv
	python3 $< --outdir results --indir results

bold: scripts/bold.py results/coherent.csv results/incoherent.csv results/nonadap.csv
	python3 $< --outdir results --indir results --dts 50