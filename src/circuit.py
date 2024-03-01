from pyrates.frontend import OperatorTemplate,NodeTemplate,CircuitTemplate,EdgeTemplate
import numpy as np
from copy import deepcopy

def create_exc_neurons(num_mt = 32, ang_coh = -90, ang_plaid1 = -45, ang_plaid2 = 180+45, 
	bandwidth = 180, num_stim_dirs = 3, div_adap = 2.0, sub_adap = 0.0, base_rate = 0.1, sat = 0.5,
	tau = 50, tau_h = 2000, inh_process = 'linear'):
	"""
	Creates pyrates-readable nodes representing excitatory motion direction-selective MT neurons.

	Arguments:
		num_mt (int): number of MT neurons, each sensitive to one direction
		ang_coh (float): angle of coherent motion direction, in degrees
		ang_plaid1 (float): angle of first incoherent motion direction, in degrees
		ang_plaid2 (float): angle of second incoherent motion direction, in degrees
		bandwidth (float): factor that determines bandwidth of receptive field
		num_stim_dirs (int): number of stimulated directions
		div_adap (float): strength of divisive adaptation
		sub_adap (float): strength of subtractive adaptation
		base_rate (float): baseline spontaneous activity of neurons
		sat (float): saturation constant of neurons
		tau (float): time constant of rate process
		tau_h (float): time constant of adaptation process
		inh_process (str): mathematical power of the divisive inhibition process (square or linear)

	Returns:
		dict: dictionary of nodes defined by pyrates operators
		list: list of edges connecting the two pyrates operators at each node
	"""

	if num_stim_dirs == 3:
		ang1 = np.pi/180 * _positive_angle(ang_plaid1)
		ang2 = np.pi/180 * _positive_angle(ang_coh)
		ang3 = np.pi/180 * _positive_angle(ang_plaid2)
		rec_field = _create_rec_field(num_stim_dirs,[ang1,ang2,ang3],bandwidth)
	elif num_stim_dirs == 4:
		ang1 = np.pi/180 * _positive_angle(ang_plaid1)
		ang2 = np.pi/180 * _positive_angle(ang_coh)
		ang3 = np.pi/180 * _positive_angle(ang_plaid2)
		ang4 = np.pi/180 * _positive_angle(ang_coh+180)
		rec_field = _create_rec_field(num_stim_dirs,[ang1,ang2,ang3,ang4],bandwidth)
	else:
		raise NotImplementedError("Only 3 or 4 stimulations directions supported at this time.")

	rate_op = _create_rate_adap_operator(inh_process,div_adap,sub_adap,base_rate,sat,tau,tau_h)

	pref_angles = np.pi/180 * np.linspace(0,360-360/num_mt,num_mt)

	neuron_names = []
	for neuron_num in range(1,num_mt+1):
		neuron_names.append(f'MTN{neuron_num}')

	nodes_dict = {}
	edges_list = []
	for n,ang in zip(neuron_names,pref_angles):
		# create nodes
		rec_field_op = deepcopy(rec_field).update_template(
			name='REC_FIELD'+n[3:],path=None,
			variables={'ap': ang})
		node = NodeTemplate(name='MTNEURON'+n[3:],path=None,
		operators=[rec_field_op,rate_op])
		nodes_dict[n] = node
		# create edges
		edge_outloc = n+'/REC_FIELD'+n[3:]+'/Resp'
		edge_inloc  = n+'/RATE_ADAP/Sens'
		edge = (edge_outloc,edge_inloc,None,{'weight': 1.0})
		edges_list.append(edge)
	return nodes_dict,edges_list


def create_inh_connections(nodes,type_edge = 'direct',type_inh = 'lateral',
	div_inib = 1.0, sub_inib = 0.0, tau = 50, delay = 0):
	"""
	Creates pyrates-readable nodes/edges representing inhibition between MT neurons.

	Arguments:
		nodes (dict): node dictionary representing excitatory MT neurons
		type_edge (str): type of inhibition (direct = new edges between neurons, interneurons = new nodes and edges)
		type_inh (str): spatial configuration of inhibitory connections (lateral, all_to_all, graded_opponent, graded_lateral)
		div_inib (float): strength of divisive inhibition
		sub_inib (float): strength of subtractive inhibition
		delay (int): number of time steps to delay differential equations

	Returns:
		dict: dictionary of new nodes (interneurons or empty)
		list: list of new edges
	"""
	nodes_dict = {}
	edges_list = []
	if type_edge == 'direct':
		edges_list = _create_direct_inhibition(nodes,type_inh,div_inib,sub_inib,delay)
		return nodes_dict,edges_list
	else:
		raise NotImplementedError("Inhibition edge can only be direct.")


def _create_direct_inhibition(nodes,type_inh,div_inib,sub_inib,delay):
	"""
	Creates pyrates-readable edges representing direct inhibition between MT neurons.

	Arguments:
		nodes (dict): node dictionary representing excitatory MT neurons
		type_inh (str): spatial configuration of inhibitory connections (lateral, all_to_all, graded_opponent, graded_lateral)
		div_inib (float): strength of divisive inhibition
		sub_inib (float): strength of subtractive inhibition
		delay (int): number of time steps to delay differential equations

	Returns:
		list: list of new edges
	"""
	edges = []
	neurons = list(nodes.keys())
	if type_inh in ['all_to_all','graded_opponent','graded_lateral']:
		for i in range(0,len(neurons)):
			for j in range(i+1,len(neurons)):
				n1 = neurons[i]
				n2 = neurons[j]
				if type_inh == 'all_to_all':
					weight = 1.0
				elif type_inh == 'graded_opponent':
					a1 = nodes[n1][f'REC_FIELD{n1[3:]}']['ap']
					a2 = nodes[n2][f'REC_FIELD{n2[3:]}']['ap']
					weight = 0.5*(1-np.cos(a1-a2))
				elif type_inh == 'graded_lateral':
					a1 = nodes[n1][f'REC_FIELD{n1[3:]}']['ap']
					a2 = nodes[n2][f'REC_FIELD{n2[3:]}']['ap']
					weight = 0.5*(1+np.cos(a1-a2))
				if div_inib > 0:
					outloc1 = n1+'/RATE_ADAP/FRate'
					inloc1 = n2+'/RATE_ADAP/Inorm'
					edge1 = (outloc1,inloc1,None,{'weight': div_inib*weight,'delay': delay})
					outloc2 = n2+'/RATE_ADAP/FRate'
					inloc2 = n1+'/RATE_ADAP/Inorm'
					edge2 = (outloc2,inloc2,None,{'weight': div_inib*weight,'delay': delay})
					edges.append(edge1)
					edges.append(edge2)
				if sub_inib > 0:
					outloc1 = n1+'/RATE_ADAP/FRate'
					inloc1 = n2+'/RATE_ADAP/Inib'
					edge1 = (outloc1,inloc1,None,{'weight': sub_inib*weight,'delay': delay})
					outloc2 = n2+'/RATE_ADAP/FRate'
					inloc2 = n1+'/RATE_ADAP/Inib'
					edge2 = (outloc2,inloc2,None,{'weight': sub_inib*weight,'delay': delay})
					edges.append(edge1)
					edges.append(edge2)
	elif type_inh == 'lateral':
		for i in range(0,len(neurons)):
			n1 = neurons[i]
			n2 = neurons[min(i+1,i+1-len(neurons))]
			n3 = neurons[i-1]
			if div_inib > 0:
				outloc1 = n1+'/RATE_ADAP/FRate'
				inloc1 = n2+'/RATE_ADAP/Inorm'
				edge1 = (outloc1,inloc1,None,{'weight': div_inib,'delay': delay})
				outloc2 = n1+'/RATE_ADAP/FRate'
				inloc2 = n3+'/RATE_ADAP/Inorm'
				edge2 = (outloc2,inloc2,None,{'weight': div_inib,'delay': delay})
				edges.append(edge1)
				edges.append(edge2)
			if sub_inib > 0:
				outloc1 = n1+'/RATE_ADAP/FRate'
				inloc1 = n2+'/RATE_ADAP/Inib'
				edge1 = (outloc1,inloc1,None,{'weight': sub_inib,'delay': delay})
				outloc2 = n1+'/RATE_ADAP/FRate'
				inloc2 = n3+'/RATE_ADAP/Inib'
				edge2 = (outloc2,inloc2,None,{'weight': sub_inib,'delay': delay})
				edges.append(edge1)
				edges.append(edge2)
	elif type_inh == 'opponent':
		for i in range(0,len(neurons)):
			n1 = neurons[i]
			n2 = neurons[min(i+int(0.5*len(neurons)),i-int(0.5*len(neurons)))]
			if div_inib > 0:
				outloc1 = n1+'/RATE_ADAP/FRate'
				inloc1 = n2+'/RATE_ADAP/Inorm'
				edge1 = (outloc1,inloc1,None,{'weight': div_inib,'delay': delay})
				edges.append(edge1)
			if sub_inib > 0:
				outloc1 = n1+'/RATE_ADAP/FRate'
				inloc1 = n2+'/RATE_ADAP/Inorm'
				edge1 = (outloc1,inloc1,None,{'weight': sub_inib,'delay': delay})
				edges.append(edge1)
	else:
		raise NotImplementedError("Direct inhibition can only be all-to-all, lateral, opponent, graded lateral or graded with stronger opponency.")
	return edges


def _create_rate_operator(sat,tau):
	"""
	Creates pyrates-readable operator representing firing rate of interneurons.

	Arguments:
		sat (float): saturation constant of interneurons
		tau (float): time constant of rate process for interneurons

	Returns:
		operator: instance of pyrates operator template with firing rate equations
	"""
	RATE = OperatorTemplate(name='RATE',path=None,
		equations=[
			'Irect = maximum(Exc,Imin)',
			'd/dt * FRate = -FRate/tau + Irect^2/(tau * (sat^2 + Irect^2))'],
		variables={
			'Irect': 'variable',
			'Exc': 'input',
			'Imin': 0.0,
			'FRate': 'output',
			'tau': tau,
			'sat': sat},
		description='potential-to-rate operator for interneurons')
	return RATE


def _create_rate_adap_operator(inh_process, DivAdapStr, SubAdapStr, IBase, sat, tau, tau_h):
	"""
	Creates pyrates-readable operator representing firing rate of adaptive neurons.

	Arguments:
		inh_process (str): type of inhibitory process (linear or square)
		DivAdapStr (float): strength of divisive adaptation
		SubAdapStr (float): strength of subtractive adaptation
		Ibase (float): baseline spontaneous activity of neurons
		sat (float): saturation constant of neurons
		tau (float): time constant of rate process
		tau_h (float): time constant of adaptation process

	Returns:
		operator: instance of pyrates operator template with adaptive firing rate equations
	"""
	if inh_process == 'square':
		RATE_ADAP = OperatorTemplate(name='RATE_ADAP',path=None,
			equations=[
				'ISyn = Sens - SubAdapStr * Adap - Inib + Exc + IBase',
				'Irect = maximum(ISyn,Imin)',
				'd/dt * FRate = -FRate/tau + Irect^2/(tau * (sat^2 + DivAdapStr * Adap + Irect^2 + Inorm^2))',
				'd/dt * Adap = (-Adap + FRate)/tau_h'],
			variables={
				'ISyn': 'variable',
				'Sens': 'input',
		    	'DivAdapStr': DivAdapStr,
		    	'SubAdapStr': SubAdapStr,
		    	'Adap': 'variable',
		    	'Inib': 'input',
		    	'Exc': 'input',
		    	'IBase': IBase,
		    	'Irect': 'variable',
		    	'Inorm': 'input',
		    	'Imin': 0.0,
		    	'FRate': 'output',
		    	'sat': sat,
		    	'tau': tau,
		    	'tau_h': tau_h},
			description="potential-to-rate operator with adaptation")
		return RATE_ADAP
	if inh_process == 'linear':
		RATE_ADAP = OperatorTemplate(name='RATE_ADAP',path=None,
			equations=[
				'ISyn = Sens - SubAdapStr * Adap - Inib + Exc + IBase',
				'Irect = maximum(ISyn,Imin)',
				'd/dt * FRate = -FRate/tau + Irect^2/(tau * (sat^2 + DivAdapStr * Adap + Irect^2 + Inorm))',
				'd/dt * Adap = (-Adap + FRate)/tau_h'],
			variables={
				'ISyn': 'variable',
				'Sens': 'input',
		    	'DivAdapStr': DivAdapStr,
		    	'SubAdapStr': SubAdapStr,
		    	'Adap': 'variable',
		    	'Inib': 'input',
		    	'Exc': 'input',
		    	'IBase': IBase,
			   	'Irect': 'variable',
			   	'Inorm': 'input',
			   	'Imin': 0.0,
			   	'FRate': 'output',
			   	'sat': sat,
			   	'tau': tau,
			   	'tau_h': tau_h},
			description="potential-to-rate operator with adaptation")
		return RATE_ADAP
	else:
		raise NotImplementedError("Only square and linear inhibition processes implemented.")

def _create_rec_field(num_stim_dirs, stim_angles, bandwidth):
	"""
	Creates pyrates-readable operator representing receptive field of motion-sensitive neurons.

	Arguments:
		num_stim_dirs (int): number of directions stimulated during experiment
		stim_angles (list): list of directions, in angles, of coherent and incoherent directions
							[ang_plaid1,ang_coherent,ang_plaid2]
		bandwidth (float): factor that determines bandwidth of receptive field
		
	Returns:
		operator: instance of pyrates operator template with preferred direction of motion ap
	"""
	if num_stim_dirs == 3:
		[a1,a2,a3] = stim_angles
		REC_FIELD = OperatorTemplate(name='REC_FIELD',path=None,
			equations=[
				'Resp = c1 * exp(k * (cos(a1-ap) - 1)) + c2 * exp(k * (cos(a2-ap) - 1)) + c3 * exp(k * (cos(a3-ap) - 1))'],
			variables={
				'Resp': 'output',
				'k': bandwidth,
				'c1': 0.,
				'c2': 0.,
				'c3': 0.,
				'a1': a1,
				'a2': a2,
				'a3': a3,
				'ap': 0.},
			description='receptive field of MT cell tuned to a preferred direction of motion')
		return REC_FIELD
	elif num_stim_dirs == 4:
		[a1,a2,a3,a4] = stim_angles
		REC_FIELD = OperatorTemplate(name='REC_FIELD',path=None,
			equations=[
				'Resp = c1 * exp(k * (cos(a1-ap) - 1)) + c2 * exp(k * (cos(a2-ap) - 1)) + c3 * exp(k * (cos(a3-ap) - 1)) + c4 * exp(k * (cos(a4-ap) - 1))'],
			variables={
				'Resp': 'output',
				'k': bandwidth,
				'c1': 0.,
				'c2': 0.,
				'c3': 0.,
				'c4': 0.,
				'a1': a1,
				'a2': a2,
				'a3': a3,
				'a4': a4,
				'ap': 0.},
			description='receptive field of MT cell tuned to a preferred direction of motion')
		return REC_FIELD

	else:
		raise NotImplementedError("Only 3 stimulations directions supported at this time.")


def _positive_angle(ang):
	"""
	Corrects non-positive angles and ensures that angles are in degrees, not radians.

	Arguments:
		ang (float): angle

	Returns:
		float: positive angle in interval [0,360)
	"""
	assert len(str(ang-int(ang))) < 6, "Angles should be in degrees, not radians."
	return (ang+360)%360

