from src.bold import calc_bold_signal
from src.plots import plot_bold,plot_activity_quadrants,plot_activity_plaid_grating,plot_interneurons,plot_total_activity
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

	# Plot simulated data
	plot_activity_plaid_grating('coherent',df_coh,args.outdir,args.grating,args.plaid1,args.plaid2,args.n)
	plot_activity_plaid_grating('incoherent',df_incoh,args.outdir,args.grating,args.plaid1,args.plaid2,args.n)
	plot_activity_plaid_grating('nonadap',df_nonadap,args.outdir,args.grating,args.plaid1,args.plaid2,args.n)

	plot_activity_quadrants('coherent',df_coh,args.n,args.outdir)
	plot_activity_quadrants('incoherent',df_incoh,args.n,args.outdir)
	plot_activity_quadrants('nonadap',df_nonadap,args.n,args.outdir)

	plot_total_activity(df_coh,df_incoh,df_nonadap,args.outdir)

	if len(df_coh.columns) > args.n:
		plot_interneurons('coherent',df_coh,args.n,args.outdir)
		plot_interneurons('incoherent',df_coh,args.n,args.outdir)
		plot_interneurons('nonadap',df_coh,args.n,args.outdir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Simulate activity of motion-sensitive adaptive neurons with inhibitory connections")
	
	parser.add_argument("--n", type=int, default=32, help="Number of MT neurons simulated")
	parser.add_argument("--grating", type=float, default=-90, help="Angle of coherent motion (in degrees)")
	parser.add_argument("--plaid1", type=float, default=180, help="Angle of first plaid (in degrees)")
	parser.add_argument("--plaid2", type=float, default=0, help="Angle of second plaid (in degrees)")	
	parser.add_argument("--outdir",required=True,help="Output directory to save bold data and plots")
	parser.add_argument("--indir",required=True,help="Directory with simulated data")

	#parser.add_argument("",type=,default=,help="")

	args = parser.parse_args()
	main(args)

