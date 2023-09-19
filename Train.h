
// functions for training and testing a net on the gpu

void forward_net(Net *);
void backward_net(Net *);
__global__ void forward_net_sub(Net *, int);
__global__ void prep_net(Net *net, float *, float *);
__global__ void update_net(Net *, float);
__global__ void backward_net_sub(Net *net, const int start);



void train(Net *net, SampleLoader loader, int n_training, float learning_rate, int minibatch_size) {
	// uses n_training samples to perform n_training/minibatch_size steps of minibatch gradient descent 
	for (int i = 0; i < n_training; ++i) {
		sample s = loader.load_training_sample();
		prep_net<<<N_BLOCKS, N_THREADS>>>(net, s.first, s.second);
   		forward_net(net);
		backward_net(net);
		if (i % minibatch_size == 0) {
			update_net<<<N_BLOCKS, N_THREADS>>>(net, learning_rate);
		}
		cudaDeviceSynchronize();
	}
}

float test(Net *net, SampleLoader loader) {
	// uses all test samples of the SampleLoader, checks if net gives correct output on them and returns accuracy
	sample *samples = loader.load_test_samples();

	int correct_count = 0;

	for (int i = 0; i < loader.n_test; ++i) {
		const sample s = samples[i];
		prep_net<<<N_BLOCKS, N_THREADS>>>(net, s.first, s.second);
   		forward_net(net);
   		cudaDeviceSynchronize();

   		Node *out_nodes = net->nodes + net->n_inp + net->n_hid;
   		float max_out = out_nodes[0].act;
   		int max_index = 0;
   		for (int j = 1; j < net->n_out; ++j) {
   			if (out_nodes[j].act > max_out) {
   				max_out = out_nodes[j].act;
   				max_index = j;
   			}
   		}

   		if (s.second[max_index] > 0.5) {
   			// means it is one, thus classified sample correctly
   			correct_count ++;
   		}
	}
	return (float)correct_count / loader.n_test; 
}


__global__ void update_net(Net *net, float learning_rate) {
	// updates all parameters (weights and biases) with their gradients and sets the gradients back to 0
	// every block takes one node, every thread takes one weight at a time ending at that node
	for (int i = net->n_inp + blockIdx.x; i < net->n_tot; i += N_BLOCKS) {
		Node *nd = net->nodes + i;
		update_node(nd, learning_rate);
	}
}


// the forward pass is calculated by taking every node except input nodes and performing forward passes on them
// each block takes one node and thus N_BLOCKS nodes are calculated at once
void forward_net(Net *net) {
	// hidden nodes:
	for (int i = 0; i < (net->n_hid + N_BLOCKS - 1) / N_BLOCKS; ++i) {
		forward_net_sub<<<N_BLOCKS, N_THREADS>>>(net, i * N_BLOCKS);
	}
	// now the output nodes
	forward_net_sub<<<net->n_out, N_THREADS>>>(net, net->n_hid);
	// and the loss node
	forward_net_sub<<<1, N_THREADS>>>(net, net->n_hid + net->n_out);
}

__global__ void forward_net_sub(Net *net, const int start) {
	int nd_index = blockIdx.x + net->n_inp + start;
	if (nd_index < net->n_tot) {
		forward_node(net->nodes + nd_index);
	}
}

// the backward pass works in the same way, but backwards (i.e. working backwards in blocks)
void backward_net(Net *net) {
	// loss node
	backward_net_sub<<<1, N_THREADS>>>(net, net->n_hid + net->n_out);
	// output nodes
	backward_net_sub<<<net->n_out, N_THREADS>>>(net, net->n_hid);
	// hidden nodes
	for (int i = (net->n_hid + N_BLOCKS - 1) / N_BLOCKS - 1; i >= 0; --i) {
		backward_net_sub<<<N_BLOCKS, N_THREADS>>>(net, i * N_BLOCKS);
	}
}

__global__ void backward_net_sub(Net *net, const int start) {
	int nd_index = blockIdx.x + net->n_inp + start;
	if (nd_index < net->n_tot) {
		backward_node(net->nodes + nd_index);
	}
}


__global__ void prep_net(Net *net, float *inp_d, float *outp_d) {
	// prepares the net for doing a forward and then a backward pass
	int t = threadIdx.x;
	int b = blockIdx.x;
	int id = t + b * N_THREADS;
	int totalthreads = N_THREADS * N_BLOCKS;
	// set gradient of loss node to 1
	if (t == 0 and b == 0) {
		net->nodes[net->n_tot - 1].grad = 1;
	}
	// now every thread takes one node to set input
	for (int index = id; index < net->n_inp; index += totalthreads) {
		net->nodes[index].act = inp_d[index];
	}
	// and every thread takes one output weight to store the correct outputs to, parallelization only really useful for really large n_out
	for (int index = id; index < net->n_out; index += totalthreads) {
		net->nodes[net->n_tot - 1].in[index].wt = outp_d[index];
	}
}

// the restructuring works by assigning a restructure score to each weight and removing those with a low score (indicating that the weight is less useful)
// depending on preference the restructure score can be calculated while training (i.e. while performing a number of SGD steps) or the parameters
// can be fixed while doing forward and backward passes and calculating the restructure score. the first option offers better performance, the second one
// generally a more useful score as the restructure score is calculated with the latest parameters 

float restructure_score(const Weight &w) {
	// calculates a score >= 0 for a weight that will be used to calculate the probability of removing that weight
	// higher score means lower probability of being removed, i.e. useless or bad weights should get a low score 
	// uncomment the appropriate method

	// // use abs_grad, the average of the absolute value of the weight's gradient
	// return w.abs_grad;

	// // use absolute value of the weight itself (which makes prep_for_restructuring unnecessary)
	// return abs(w.wt);

	// no idea if this one works better than the other two but intuitively it makes most sense to me
	return abs(w.wt) * w.abs_grad;
}


void train_and_restructure(Net *net, SampleLoader loader, int n_add, int n_samples, int n_batch, float learning_rate) {
	// trains the network with n_samples iterations and n_batch batchsize, then removes around n_add unimportant weights and 
	// adds new random ones initialized to zero

	train(net, loader, n_samples, learning_rate, n_batch);

	// the rest of this function is for the restructuring. the idea is to remove the n_add weights with the lowest restructure score. 
	// this is done by estimating the cutoff score for removing weights and then removing the ones with a score below that.
	// that is very inelegant cause instead of n_add weights some other number of weights is replaced. 
	// to do this properly one would have to sort the weights by their restructure score, which is probably a bit inefficient for large networks,
	// or use a method for finding the k-th smallest element of an array in linear time, but i'm too lazy to implement it.

	int num_estimation_weights = 5 * net->n_tot; // should be more than number of nodes to avoid selecting too few weights...
	// collect the restructure_score of roughly num_estimation_weights weights in a vector
	float perc = (float)num_estimation_weights / net->n_weights;
	std::vector<float> estimation_scores;
	for (int i = net->n_inp; i < net->n_tot - 1; ++i) {
		for (int j = 0; j < net->nodes[i].n_in * perc; ++j) {
			// now add correct number of scores of randomly selected weights from this node to estimation_scores
			estimation_scores.push_back(restructure_score(net->nodes[i].in[rand() % net->nodes[i].n_in]));
		}
	}
	// sort this vector
	std::sort(estimation_scores.begin(), estimation_scores.end());
	// now get estimation of value s such that correct proportion of weights gets removed if all weigts with score below s get removed
	float cutoff = estimation_scores[n_add * estimation_scores.size() / net->n_weights];

	// create new weight vectors
	std::vector<Weight> *new_wts = new std::vector<Weight>[net->n_tot]();
	// add all weights that have score higher than cutoff and count removed ones
	int removed_cnt = 0;
	for (int i = net->n_inp; i < net->n_tot - 1; ++i) {
		for (int j = 0; j < net->nodes[i].n_in; ++j) {
			if (restructure_score(net->nodes[i].in[j]) > cutoff) {
				new_wts[i].push_back(net->nodes[i].in[j]);
			}
			else removed_cnt++;
		}
	}
	// now add removed_cnt new weights, they will have value zero (could initalize somehow but i think zero-initialization makes most sense) 
	for (int i = 0; i < removed_cnt; ++i) {
		std::pair<int, int> start_end = net->random_weight_indices();
		new_wts[start_end.second].push_back(Weight(net->nodes + start_end.first, start_end.first));
	}

	// replace weight arrays with new weight arrays
	for (int i = net->n_inp; i < net->n_tot - 1; ++i) {
		gpuErrchk(cudaFree(net->nodes[i].in));
		int n_in = new_wts[i].size();
		net->nodes[i].n_in = n_in;
		gpuErrchk(cudaMallocManaged(&net->nodes[i].in, sizeof(Weight) * n_in));
		std::memcpy(net->nodes[i].in, new_wts[i].data(), sizeof(Weight) * n_in);
	}

	delete [] new_wts;

	// update with learning rate 0 to reset the stuff used for restructuring
	update_net<<<N_BLOCKS, N_THREADS>>>(net, 0);

	std::cout << "restructuring, replaced " << removed_cnt << " weights\n";
}
 