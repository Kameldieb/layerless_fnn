
// defines all the node classes and their parent class Node

// node types:
#define TANH 0
#define FAST_TANH 1
#define RELU 2
#define ID 3
#define INPUT 4
#define CELOSS 5
#define LEAKY_RELU 6

// for leaky relu
#define LEAKY_FACTOR 0.1


class Node {
	// one class for all nodes makes some things easier, but really should use inheritance or something...
public:
	int n_in = 0;
	Weight *in;		// stores incoming weights on device
	float sum;
	float act;		// activation
	float grad = 0;
	float bias_grad = 0;	// accumulating
	int type;
	float bias = 0;

	Node(char type) {
		this->type = type;
	}
	Node() {};

};

__device__ float sum_weighted_inputs(Node *n, float &res) {
	// first every thread sums its part of the weighted inputs, but waits for incoming nodes to finish their forward pass
	int id = threadIdx.x;
	__shared__ float thread_sums[N_THREADS];
	thread_sums[id] = 0;
	for (int i = id; i < n->n_in; i += blockDim.x) {
		Node *in_node = n->in[i].in;
		// wait for incoming node to finish calculation
		thread_sums[id] += n->in[i].wt * in_node->act;
	}
	__syncthreads();
	// now sum array
	// number of threads has to be power of 2
	for (int s = blockDim.x/2; s > 0; s /= 2) {
		if (id < s) {
			thread_sums[id] += thread_sums[id + s];
		}
		__syncthreads();
	}
	if (id == 0) {
		res = thread_sums[0];
	}
}


__device__ float apply_activation_function(float x, int type) {
	switch(type) {
	case RELU:
		return x > 0 ? x : 0;
		break;
	case ID:
		return x;
		break;
	case FAST_TANH:
		return x / (1 + abs(x));
		break;
	case LEAKY_RELU:
		return x > 0 ? x : x * LEAKY_FACTOR;
		break;
	default:
		assert(0);
	}
}

__device__ float apply_activation_function_derivative(float x, int type) {
	switch (type) {
	case RELU:
		return x > 0 ? 1 : 0;
		break;
	case ID:
		return 1;
		break;
	case FAST_TANH:
		return 1/((1+abs(x))*(1+abs(x)));
		break;
	case LEAKY_RELU:
		return x > 0 ? 1 : LEAKY_FACTOR;
		break;
	default:
		assert(0);
	}
}


__device__ void forward_node(Node *n) {
	// performs forward pass for node using one block (and all its threads)
	int id = threadIdx.x;
	switch (n->type) {
		// nodes that need summation and activation function
	case RELU: case ID: case FAST_TANH: case LEAKY_RELU:
		__shared__ float sum;
		sum_weighted_inputs(n, sum);
		__syncthreads();
		if (id == 0) {
			n->sum = sum + n->bias;
			n->act = apply_activation_function(n->sum, n->type);
		}
		break;
	case INPUT:
		// should never be reached as forward pass not evaluated for input nodes
		break;

	case CELOSS:
		// the loss node needs the correct label (or generally the correct outputs), but can't access it in the forward pass
		// however it doesn't need the weights from the output nodes, so the weights (just the float values) are replaced with 
		// the correct outputs. confusing but prevents further complexity (i think).
		// The network has no softmax layer, so cross entropy loss is calculated directly from the output values
		// first calculate the sum of the exponentiated output node activations
		// use serial atomicadd operations, as output size is usually small this shouldn't be too bad
		__shared__ float exp_sum;
		__shared__ float log_exp_sum;
		__shared__ float loss;
		__shared__ int max_act;	// maximum activation of the output nodes, but rounded down, only used for numerical stability
		// initialize the summing values
		if (id == 0) {
			exp_sum = 0;
			loss = 0;
			max_act = n->in[0].in->act;
		}
		// calculate max output activation for numerical stability
		__syncthreads();
		for (int i = id; i < n->n_in; i += N_THREADS) {
			atomicMax(&max_act, (int) (n->in[i].in->act));
		}
		// now exp_sum will be calculated to be the sum of exp(a_i - max_act)
		__syncthreads();
		for (int i = id; i < n->n_in; i += N_THREADS) {
			atomicAdd(&exp_sum, exp(n->in[i].in->act - max_act));
		}
		__syncthreads();

		if (id == 0) {
			log_exp_sum = log(exp_sum);
			// now store the exp_sum in the bias of the node and the max_act in the sum of the node,
			// both wouldn't be used otherwise, so the values can be reused in backward pass
			n->bias = exp_sum;
			n->sum = (float) max_act;
		}
		__syncthreads();
		for (int i = id; i < n->n_in; i += N_THREADS) {
			atomicAdd(&loss, - n->in[i].wt * (n->in[i].in->act - max_act - log_exp_sum));
		}
		__syncthreads();
		if (id == 0) {
			n->act = loss;
		}
		// now loss is done calculating
		break;
	default:
		assert(0);	// should throw error?
	}
	__syncthreads();
}

__device__ void update_node(Node *n, float learning_rate) {
	// updates the nodes and the node's incoming weights with their respective gradients and sets those gradients to zero
	if (threadIdx.x == 0) {
		n->bias -= learning_rate * n->bias_grad;
		n->bias_grad = 0;
	}
	for (int j = threadIdx.x; j < n->n_in; j += N_THREADS) {
		n->in[j].wt -= learning_rate * n->in[j].grad;
		n->in[j].grad = 0;
	}
}


__device__ void backward_node(Node *n) {
	// first wait for all later nodes to add their gradients to this node:
	int id = threadIdx.x;

	switch (n->type) {
	case RELU: case ID: case FAST_TANH: case LEAKY_RELU:
		__shared__ float bgrad;
		if (id == 0) {
			bgrad = apply_activation_function_derivative(n->sum, n->type) * n->grad;
			n->bias_grad += bgrad;
		}
		__syncthreads();
		// for nodes that sum their weighted inputs and apply activation function
		for (int i = id; i < n->n_in; i += N_THREADS) {
			// set incoming node's gradient 
			n->in[i].in->grad += n->in[i].wt * bgrad;
			// add to incoming weight's gradient
			float wt_grd = bgrad * n->in[i].in->act;
			atomicAdd(&n->in[i].grad, wt_grd);
			atomicAdd(&n->in[i].abs_grad, abs(wt_grd));
		}
		break;
	case CELOSS:
		// careful: as the loss node does not need the weight values but needs the correct label/outputs, i chose to store the correct outputs in the weight values
		// also no softmax layer is used, so this node calculates the loss directly from the outputs
		// also the sum of the exponentials is stored in the loss node's bias, which wouldn't be used otherwise, and the maximum activation of outputs in the sum
		for (int i = id; i < n->n_in; i += N_THREADS) {
			atomicAdd(&n->in[i].in->grad, exp(n->in[i].in->act - n->sum) / n->bias - n->in[i].wt);
		}
		break;
	default:
		assert(0);
	}
	// now set the node's gradient to zero
	if (id == 0) {
		n->grad = 0;
	}
}

