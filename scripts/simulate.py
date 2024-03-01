from src.circuit import create_exc_neurons,create_inh_connections
from src.stimulus import create_experimental_stimulus
from pyrates.frontend import CircuitTemplate
import argparse

def main(args):

	# Print parameters
	params = vars(args)
	f = open(f'{args.outdir}/parameters.txt','w')
	for p in params.keys():
		line = f"{p}: \t {params[p]}"
		f.write(line+'\n')
	f.close()

	# Build circuit
	nodes,edges1 = create_exc_neurons(args.n,args.grating,args.plaid1,args.plaid2,args.width,
		args.stimdirs,args.divadap,args.subadap,args.base,args.sat,args.tau,args.tauh)
	nodes2,edges2 = create_inh_connections(nodes,args.inconnect,args.inspace,args.divinib,
		args.subinib,args.delay)
	nodes.update(nodes2)
	edges = edges1 + edges2
	circuit = CircuitTemplate(name='CIRCUIT',path=None,nodes=nodes,edges=edges)

	# Define simulation inputs
	stimulus = create_experimental_stimulus(args.condition,args.dt,args.cohgra,args.cohpla,
		args.incohgra,args.incohpla,args.teststim,args.stimdirs,args.T)
	inputs = {}
	if args.stimdirs == 3:
		for cell in circuit.nodes.keys():
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c1'] = stimulus[0,:]
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c2'] = stimulus[1,:]
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c3'] = stimulus[2,:]
	elif args.stimdirs == 4:
		for cell in circuit.nodes.keys():
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c1'] = stimulus[0,:]
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c2'] = stimulus[1,:]
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c3'] = stimulus[2,:]
			inputs[f'{cell}/REC_FIELD{cell[3:]}/c4'] = stimulus[3,:]
	else: 
		raise NotImplementedError("Number of stimulated directions not supported.")

	# Define simulation outputs
	outputs={}
	for cell in circuit.nodes.keys():
		outputs[cell] = f'{cell}/RATE_ADAP/FRate'

	# Run each simulation and export data
	results = circuit.run(simulation_time=(args.T+args.dt),step_size=args.dt,
		sampling_step_size=args.dts,inputs=inputs,outputs=outputs,
		backend='default',solver='scipy')
	results.to_csv(f'{args.outdir}/{args.condition}.csv')

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Simulate activity of motion-sensitive adaptive neurons with inhibitory connections")
	
	parser.add_argument("--n", type=int, default=32, help="Number of MT neurons to simulate")
	parser.add_argument("--grating", type=float, default=-90, help="Angle of coherent motion (in degrees)")
	parser.add_argument("--plaid1", type=float, default=180, help="Angle of first plaid (in degrees)")
	parser.add_argument("--plaid2", type=float, default=0, help="Angle of second plaid (in degrees)")
	parser.add_argument("--width", type=float, default=180, help="Scalar bandwidth of receptive field")
	parser.add_argument("--stimdirs", type=int, default=4, help="Number of motion directions in stimulus")
	parser.add_argument("--divadap", type=float, default=0, help="Strength of divisive adaptation")
	parser.add_argument("--subadap", type=float, default=0, help="Strength of subtractive adaptation")
	parser.add_argument("--base", type=float, default=0.1, help="Baseline activity of MT neurons")
	parser.add_argument("--sat", type=float, default=0.5, help="Saturation constant of MT neurons")
	parser.add_argument("--tau", type=float, default=50, help="Time constant of MT neurons")
	parser.add_argument("--tauh", type=float, default=2000, help="Time constant of adaptation")
	parser.add_argument("--inconnect", default='direct', help="Type of inhibitory connections (direct, interneurons, direct_disinhibition)")
	parser.add_argument("--inspace", default='graded_opponent', help="Spatial type of inhibition (all_to_all, lateral, graded_opponent(_tau), graded_lateral(_tau))")
	parser.add_argument("--divinib", type=float, default=0, help="Strength of divisive inhibition")
	parser.add_argument("--subinib", type=float, default=0, help="Strength of subtractive inhibition")
	parser.add_argument("--delay", type=float, default=0, help="Delay time steps for integration")
	parser.add_argument("--dt", type=float, default=0.1, help="Time step for integration")
	parser.add_argument("--cohgra",type=float,default=1.0,help="Stimulus strength for coherent direction during coherent stimulus")
	parser.add_argument("--cohpla",type=float,default=0.0,help="Stimulus strength for incoherent direction during coherent stimulus")
	parser.add_argument("--incohgra",type=float,default=2.0/3,help="Stimulus strength for coherent direction during incoherent stimulus")
	parser.add_argument("--incohpla",type=float,default=1.0/6,help="Stimulus strength for incoherent direction during incoherent stimulus")
	parser.add_argument("--teststim",default='NullMov',help="Stimulus for test condition (NullMov, Coh, Incoh)")
	parser.add_argument("--T",type=float,default=48000,help="Total integration time")
	parser.add_argument("--dts",type=float,default=50,help="Sampling time step for export data")
	parser.add_argument("--outdir",required=True,help="Output directory to save simulated data")
	parser.add_argument("--condition",required=True,help="Stimulus condition (coherent, incoherent, nonadap)")

	#parser.add_argument("",type=,default=,help="")

	args = parser.parse_args()
	main(args)

