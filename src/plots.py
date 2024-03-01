import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from src.circuit import _positive_angle
from src.bold import calc_sum_activity

def plot_bold(bold,outdir):
	"""
	Plots bold signal and saves it to file.

	Arguments:
		bold (dataframe): bold signal for each condition (columns = conditions, rows = timesteps)
		outdir (str): output directory to save plot

	Returns:
		None
	"""
	ymax = np.max(np.max(bold))
	if ymax > 6:
		ylim = np.ceil(ymax+0.5)
	else:
		if 6-ymax > 4:
			ylim = np.ceil(ymax+0.5)
		else:
			ylim = 6
	ymin = np.minimum(0,np.floor(np.min(np.min(bold))))
	filename = f'{outdir}/bold.pdf'
	ax = _create_plot_specs(ymin,ylim,bold.index[-1],'BOLD')
	conditions = ['coherent','incoherent','nonadapting']
	labels = ['Adapt Coherent','Adapt Incoherent','Non-adapting']
	for condition,label in zip(conditions,labels):
		ax.plot(bold[condition],label=label)
	ax.legend()
	plt.savefig(filename,bbox_inches='tight')
	plt.close()
	return None


def plot_activity_quadrants(condition,activity,num_neurons,outdir):
	"""
	Plots firing rate activity of neurons per quadrant and saves it to file.

	Arguments:
		condition (str): simulated experimental condition (coherent,incoherent,nonadap)
		activity (dataframe): activity of each neuron (columns = neurons, rows = timesteps)
		num_neurons (int): number of MT neurons simulated
		outdir (str): output directory to save plot

	Returns:
		None
	"""
	assert num_neurons%4 == 0, "Number of neurons should be divisible by 4"
	nq = int(num_neurons/4)
	mtneurons = activity.columns[:num_neurons]
	q1 = mtneurons[:nq]
	q2 = mtneurons[nq:2*nq]
	q3 = mtneurons[2*nq:3*nq]
	q4 = mtneurons[3*nq:]

	plot_activity_neurons(q1,activity,condition,f'{outdir}/{condition}_q1.pdf')
	plot_activity_neurons(q2,activity,condition,f'{outdir}/{condition}_q2.pdf')
	plot_activity_neurons(q3,activity,condition,f'{outdir}/{condition}_q3.pdf')
	plot_activity_neurons(q4,activity,condition,f'{outdir}/{condition}_q4.pdf')

	return None

def plot_activity_plaid_grating(condition,activity,outdir,ang_coh,ang_plaid1,ang_plaid2,num_neurons):
	"""
	Plots firing rate activity of plaid and grating neurons and saves it to file.

	Arguments:
		condition (str): simulated experimental condition (coherent,incoherent,nonadap)
		activity (dataframe): activity of each neuron (columns = neurons, rows = timesteps)
		outdir (str): output directory to save plot
		ang_coh (float): angle of coherent motion direction, in degrees
		ang_plaid1 (float): angle of first incoherent motion direction, in degrees
		ang_plaid2 (float): angle of second incoherent motion direction, in degrees
		num_neurons (int): number of MT neurons simulated

	Returns:
		None
	"""
	ang_gr = _positive_angle(ang_coh)
	ang_p1 = _positive_angle(ang_plaid1)
	ang_p2 = _positive_angle(ang_plaid2)
	neurons = activity.columns[:num_neurons]
	# determine neuron number from angle
	min_ang = 360/num_neurons
	neuron_gra    = neurons[int(ang_gr/min_ang)]
	neuron_plaid1 = neurons[int(ang_p1/min_ang)]
	neuron_plaid2 = neurons[int(ang_p2/min_ang)]

	neurons = [neuron_gra,neuron_plaid1,neuron_plaid2]
	labels = ['Grating','Plaid 1','Plaid 2']
	plot_activity_neurons(neurons,activity,condition,f'{outdir}/{condition}_grapla.pdf',labels)
	return None

def plot_total_activity(activity_coherent,activity_incoherent,activity_nonadap,outdir):
	"""
	Plots total activity per condition and saves it to file.

	Arguments:
		activity (dataframe): activity of each neuron (columns = neurons, rows = timesteps)
		outdir (str): output directory to save plot

	Returns:
		None
	"""
	total_activity = calc_sum_activity(activity_coherent.columns,activity_coherent,activity_incoherent,activity_nonadap)
	plot_activity_neurons(total_activity.columns,total_activity,'total',f'{outdir}/total.pdf')
	return None


def plot_interneurons(condition,activity,num_neurons,outdir):
	"""
	Plots inhibition per quadrant per target neuron and saves it to file.

	Arguments:
		condition (str): simulated experimental condition (coherent,incoherent,nonadap)
		activity (dataframe): activity of each neuron (columns = neurons, rows = timesteps)
		num_neurons (int): number of MT neurons simulated
		outdir (str): output directory to save plot

	Returns:
		None
	"""
	assert num_neurons%4 == 0, "Number of neurons should be divisible by 4"
	nq = int(num_neurons/4)
	mtneurons = activity.columns[:num_neurons]
	q1 = mtneurons[:nq]
	q2 = mtneurons[nq:2*nq]
	q3 = mtneurons[2*nq:3*nq]
	q4 = mtneurons[3*nq:]

	inhibition = pd.DataFrame(columns=mtneurons,index=activity.index)
	for neuron in mtneurons:
		inhibition.loc[:,(neuron)] = activity[activity.columns[[f'_{neuron[3:]}' in IIN for IIN in activity.columns]]].sum(axis=1)


	for group,name in zip([q1,q2,q3,q4],['q1','q2','q3','q4']):
		labels = [f'Inhibition to {neurons}' for neurons in group]
		plot_activity_neurons(group,inhibition,condition,f'{outdir}/{condition}_inhibition_{name}.pdf',labels)

	return None

def plot_activity_neurons(neurons,activity,condition,filename,labels=None):
	"""
	Plots firing rate activity of group of neurons and saves it to file.

	Arguments:
		neurons (list): list of neurons to plot
		activity (dataframe): activity of each neuron (columns = neurons, rows = timesteps)
		condition (str): simulated experimental condition (coherent,incoherent,nonadap)
		filename (str): name and extension of saved plot
		labels (list): plot labels

	Returns:
		None
	"""
	ymax = np.ceil(activity.max().max())
	ylim = np.maximum(ymax,1)
	ax = _create_plot_specs(0,ylim,activity.index[-1],'Firing Rate')
	if labels == None:
		labels = neurons
	for neuron,label in zip(neurons,labels):
		ax.plot(activity[neuron],label=label)
	ax.legend()
	titles = {'coherent':'Adapt Coherent','incoherent':'Adapt Incoherent','nonadap':'Non-adapting','total':'Total Activity'}
	ax.set_title(titles[condition])
	plt.savefig(filename,bbox_inches='tight')
	plt.close()
	return None

def _create_plot_specs(ymin,ymax,T,ylabel):
	"""
	Creates and formats ax object for plotting.

	Arguments:
		ymax (float): upper bound for y axis
		T (float): maximum value for x axis (ms)
		ylabel (str): label for y axis

	Returns:
		ax: ax object with specified formatting in x and y axes
	"""
	assert T%6000 == 0, "Total time to be plotted should be a multiple of 6 seconds"
	fig,ax = plt.subplots(1,1)
	fig.set_size_inches(10,5)
	ax.set_xlim([0,T])
	ax.set_xticks(np.arange(0,T+1,6000))
	ax.set_xticklabels(1e-3*np.arange(0,T+1,6000))
	ax.set_xlabel('Time (s)')
	ax.set_ylim([ymin,ymax])
	ax.set_ylabel(ylabel)
	return ax