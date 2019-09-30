from .simple import *
from .convolution import *
from .baseop import HEADER, LINE

op_types = {
	'convolutional': convolutional,
	'conv-select': conv_select,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'sigmoid': sigmoid,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg,
	'conv-extract': conv_extract,
	'extract': extract,
	'parallel-merge': parallel_merge,
	'merge': merge
}

is_parallel_state = False
parallel_input = None
parallel_layers = []
named_layers = {}

def op_create(*args):
	global  is_parallel_state, parallel_input, parallel_layers, named_layers

	layer_type = list(args)[0].type

	# parallel structure continue
	if is_parallel_state and (layer_type == 'leaky' or layer_type == 'sigmoid'):
		l = op_types[layer_type](*args)
		parallel_layers.append(l)
		return l

	# parallel structure end
	if layer_type == 'parallel-merge':
		tensors = []
		tensors.append(parallel_input)

		for l in parallel_layers:
			tensors.append(l.out)

		new_args = [args[0], tensors, args[2], args[3], args[4], args[5]]
		l = op_types[layer_type](*new_args)

		is_parallel_state = False
		parallel_input = None
		parallel_layers = []

		return l

	if hasattr(args[0], 'is_parallel'):
		is_parallel = args[0].is_parallel

		# Parallel structure start
		if is_parallel == True :
			if not is_parallel_state :
				is_parallel_state = True
				parallel_input = args[1]

			new_args = [args[0], parallel_input, args[2], args[3], args[4], args[5]]
			l = op_types[layer_type](*new_args)
			if args[0].activation == 'linear':
				parallel_layers.append(l)
			return l

	# Normal (vertical) structure
	l = op_types[layer_type](*args)
	return l