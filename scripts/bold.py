from src.bold import calc_bold_signal
from src.plots import plot_bold,plot_activity_quadrants,plot_activity_plaid_grating
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(args):
	# Load simulation data
	df_coh = pd.read_csv(f'{args.indir}/coherent.csv',header=[0],index_col=0)
	df_incoh = pd.read_csv(f'{args.indir}/incoherent.csv',header=[0],index_col=0)
	df_nonadap = pd.read_csv(f'{args.indir}/nonadap.csv',header=[0],index_col=0)

	# Calculate BOLD and export data
	neurons = np.array(df_coh.columns)
	activity,uncorrected,bold = calc_bold_signal(neurons,df_coh,df_incoh,df_nonadap,args.T,args.dts,args.dtsbold)
	bold.to_csv(f'{args.outdir}/bold.csv')
	#uncorrected.to_csv(f'{args.outdir}/bold_uncorrected.csv')

	# Plot BOLD
	plot_bold(bold,args.outdir)

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Simulate activity of motion-sensitive adaptive neurons with inhibitory connections")
	
	parser.add_argument("--T",type=float,default=48000,help="Total integration time")
	parser.add_argument("--dts",type=float,default=10,help="Sampling time step of simulated activity")
	parser.add_argument("--dtsbold",type=float,default=100,help="Sampling time step for bold data")
	parser.add_argument("--outdir",required=True,help="Output directory to save bold data and plots")
	parser.add_argument("--indir",required=True,help="Directory with simulated data")

	#parser.add_argument("",type=,default=,help="")

	args = parser.parse_args()
	main(args)

