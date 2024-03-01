import numpy as np


def create_experimental_stimulus(condition, dt, c_coh_gra, c_coh_pla, c_incoh_gra, c_incoh_pla, 
	test = 'NullMov', num_stim_dirs = 3, T = 48000, t0 = 0, t0adap = 6000, t0test = 36000, 
	duralt = 1500):
	"""
	Creates stimulus conditions to pass to neurons, according to experimental protocol.

	Arguments:
		condition (str): experimental adapting condition (nonadap, coherent, incoherent)
		dt (float): time step of integration
		c_coh_gra (float): stimulus intensity for coherent (grating) direction during coherent
		c_coh_pla (float): stimulus intensity for incoherent (plaid) directions during coherent
		c_incoh_gra (float): stimulus intensity for coherent direction during incoherent
		c_incoh_pla (float): stimulus intensity for incoherent direction during incoherent
		test (str): test condition (NullMov, Coh, Incoh)
		num_stim_dirs (int): number of stimulated directions in nonadap
		T (float): total integration time (ms)
		t0 (float): beginning of simulation (ms)
		t0adap (float): beginning of adapting condition (ms)
		t0test (float): beginning of test period (ms)
		duralt (float): duration of alternations in nonadap
		
	Returns:
		array: stimulus intensity for each time step (columns) for each direction (rows)
	"""
	initial_contrast = _build_const_period(t0,t0adap,dt,'NullMov',num_stim_dirs,c_coh_gra,c_coh_pla,
		c_incoh_gra,c_incoh_pla)

	if condition == 'nonadap':
		adap_contrast = _build_non_adap_period(t0adap,t0test,dt,duralt,num_stim_dirs,
			c_coh_gra,c_coh_pla,c_incoh_gra,c_incoh_pla)
	elif condition == 'coherent':
		adap_contrast = _build_const_period(t0adap,t0test,dt,'Coh',num_stim_dirs,c_coh_gra,c_coh_pla,
			c_incoh_gra,c_incoh_pla)
	elif condition == 'incoherent':
		adap_contrast = _build_const_period(t0adap,t0test,dt,'Incoh',num_stim_dirs,c_coh_gra,c_coh_pla,
			c_incoh_gra,c_incoh_pla)
	else:
		raise Exception("Invalid stimulus condition.")

	test_contrast = _build_const_period(t0test,T,dt,test,num_stim_dirs,c_coh_gra,c_coh_pla,c_incoh_gra,
		c_incoh_pla)

	if num_stim_dirs in [3,4]:
		contrast = np.concatenate((initial_contrast,adap_contrast,test_contrast),axis=1)
	else:
		raise NotImplementedError("Only 3 or 4 directions supported for non-adapting condition.")
	return contrast

def _build_const_period(ti, tf, dt, typemov, num_stim_dirs, c_coh_gra, c_coh_pla, c_incoh_gra, c_incoh_pla):
	"""
	Creates constant stimulus.

	Arguments:
		ti (float): initial time
		tf (float): final time
		dt (float): time step
		typemov (str): type of movement (NullMov, Coh, Incoh)
		num_stim_dirs (int): number of stimulated directions
		c_coh_gra (float): stimulus intensity for coherent (grating) direction during Coh
		c_coh_pla (float): stimulus intensity for incoherent (plaid) directions during Coh
		c_incoh_gra (float): stimulus intensity for coherent direction during Incoh
		c_incoh_pla (float): stimulus intensity for incoherent direction during Incoh

	Returns:
		array: stimulus intensity for each time step (columns) for each direction (rows)
	"""
	dur = tf - ti
	period = np.ones(shape=(1,int(np.round(dur/dt,decimals=0))))
	contrast = _create_input_time(period,typemov,num_stim_dirs,c_coh_gra,c_coh_pla,c_incoh_gra,c_incoh_pla)
	return contrast

def _build_non_adap_period(ti, tf, dt, duralt, num_stim_dirs, c_coh_gra, c_coh_pla, 
	c_incoh_gra, c_incoh_pla):
	"""
	Creates non-adapting alternating stimulus.

	Arguments:
		ti (float): initial time
		tf (float): final time
		dt (float): time step
		duralt (float): duration of alternations
		num_stim_dirs (int): number of stimulated directions
		c_coh_gra (float): stimulus intensity for coherent (grating) direction during Coh
		c_coh_pla (float): stimulus intensity for incoherent (plaid) directions during Coh
		c_incoh_gra (float): stimulus intensity for coherent direction during Incoh
		c_incoh_pla (float): stimulus intensity for incoherent direction during Incoh

	Returns:
		array: stimulus intensity for each time step (columns) for each direction (rows)
	"""
	period = np.ones(shape=(1,int(np.round(duralt/dt,decimals=0))))
	dur = int(np.round((tf-ti)/dt,decimals=0))

	# non-adapting condition as alternated coherent and incoherent stimuli
	if num_stim_dirs == 3:
		coh_contrast = _create_input_time(period,'Coh',num_stim_dirs,c_coh_gra,c_coh_pla,c_incoh_gra,c_incoh_pla)
		incoh_contrast = _create_input_time(period,'Incoh',num_stim_dirs,c_coh_gra,c_coh_pla,c_incoh_gra,c_incoh_pla)
		contrast = coh_contrast
		while contrast.shape[1] < dur:
			contrast = np.concatenate((contrast,incoh_contrast,coh_contrast),axis=1)
		contrast = contrast[:,:dur]
		return contrast

	# non-adapting condition determined from experimental stimulus
	elif num_stim_dirs == 4:
		#assert c_coh_gra == c_incoh_gra + 2 * c_incoh_pla, "Stimulus strength not well defined"
		a = c_incoh_gra
		b = c_incoh_pla
		ab = a+2*b
		c_left  = [a,0,0,b,ab,b,0]
		c_down  = [b,0,b,a,0,0,0]
		c_right = [0,0,a,b,0,b,ab]
		c_up    = [b,ab,b,0,0,a,0]
		(c1,c2,c3,c4) = (0,ab,0,0)
		sequence = np.reshape(np.array([c1,c2,c3,c4]),(4,1)) * period
		for c1,c2,c3,c4 in zip(c_left,c_down,c_right,c_up):
			mini_seq = np.reshape(np.array([c1,c2,c3,c4]),(4,1)) * period
			sequence = np.concatenate((sequence,mini_seq),axis=1)
		contrast = sequence
		while contrast.shape[1] < dur:
			contrast = np.concatenate((contrast,sequence),axis=1)
		contrast = contrast[:,:dur]
		return contrast
	else:
		raise NotImplementedError("More than 3 directions in non-adapting condition not implemented.")

def _create_input_time(timearray, typemov, num_stim_dirs, c_coh_gra, c_coh_pla, c_incoh_gra, c_incoh_pla):
	"""
	Creates array of stimulus intensity for each time step.

	Arguments:
		timearray (array): array of time steps for duration of condition
		typemov (str): type of movement (NullMov, Coh, Incoh)
		num_stim_dirs (int): number of stimulated directions
		c_coh_gra (float): stimulus intensity for coherent (grating) direction during Coh
		c_coh_pla (float): stimulus intensity for incoherent (plaid) directions during Coh
		c_incoh_gra (float): stimulus intensity for coherent direction during Incoh
		c_incoh_pla (float): stimulus intensity for incoherent direction during Incoh

	Returns:
		array: stimulus intensity for each time step (columns) for each direction (rows)
	"""
	if typemov == 'NullMov':
		c1 = 0.0
		c2 = 0.0
		c3 = 0.0
		c4 = 0.0
	elif typemov == 'Coh':
		c1 = c_coh_pla
		c2 = c_coh_gra
		c3 = c_coh_pla
		c4 = 0.0
	elif typemov == 'Incoh':
		c1 = c_incoh_pla
		c2 = c_incoh_gra
		c3 = c_incoh_pla
		c4 = 0.0
	else:
		raise NotImplementedError("Motion other than coherent, incoherent or static not implemented.")
	if num_stim_dirs == 3:
		contrast = np.reshape(np.array([c1,c2,c3]),(3,1))
	elif num_stim_dirs == 4:
		contrast = np.reshape(np.array([c1,c2,c3,c4]),(4,1))
	else: 
		raise NotImplementedError("Number of stimulated directions must be 3 or 4.")
	return contrast * timearray









