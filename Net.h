
#define EPSILON 0.0001

// stolen somewhere
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


class Net {
// class for feed forward net without layers
// basically all nodes are in one big array and weights can connect any two nodes in a feedforward style, but
	// 1. weights can't connect input to input nodes or output to output nodes
	// 2. weights can't connect "hidden" nodes (term i use for nodes that aren't input or output) that are less than N_BLOCKS apart, as
	//    N_BLOCKS nodes are processed in parallel, so having weights between leads to errors (although likely irrelevant)
public:
	int n_inp;
	int n_hid;	// hidden compute nodes, number should be multiple of N_BLOCKS i think
	int n_out;
	int n_tot;
	int n_weights;
	Node *nodes;

	// proportion of weights starting at an input node and weights ending at an output node
	// used in the beginning and when adding new weights
	float weight_inp_perc, weight_outp_perc;

	Net(int n_inp, int n_hid, int n_out, int n_weights, float weight_inp_perc, float weight_outp_perc, int activation_function_hid=RELU) {
		// constructs a feed forward net without layers and cross entropy loss, can be used in general only for classifiers 
		// last node is the loss node, weights to the loss node are fixed
		this->n_inp = n_inp;
		this->n_out = n_out;
		this->n_hid = n_hid;
		this->n_tot = n_inp + n_hid + n_out + 1;
		this->weight_inp_perc = weight_inp_perc;
		this->weight_outp_perc = weight_outp_perc;
		this->n_weights = n_weights;

		// construct and fill array containing all the nodes
		cudaMallocManaged(&nodes, sizeof(Node) * n_tot);

		for (int i = 0; i < n_inp; ++i) {
			nodes[i] = Node(INPUT);	// activation function of input nodes is irrelevant, never called
		}
		for (int i = n_inp; i < n_inp + n_hid; ++i) {
			nodes[i] = Node(activation_function_hid);
		}
		for (int i = n_inp + n_hid; i < n_inp + n_hid + n_out; ++i) {
			nodes[i] = Node(ID);
		}
		// last node is cross entropy loss node
		nodes[n_tot - 1] = Node(CELOSS);

		// create array of incoming weights for every node except input nodes
		std::vector<Weight> *weight_lists = new std::vector<Weight>[n_tot - n_inp]();

		// create n_weights weights with random start and end nodes, add them to the correct weight array
		for (int i = 0; i < n_weights; ++i) {

			int start = random_unif() < weight_inp_perc ? rand() % n_inp : rand() % n_hid + n_inp;
			int end = random_unif() < weight_outp_perc ? rand() % n_out + n_inp + n_hid : rand() % n_hid + n_inp;
			
			if (start > end) {
				std::swap(start, end);
			}
			if ((end - n_inp) / N_BLOCKS == (start - n_inp) / N_BLOCKS and end < n_inp + n_hid) {
				// in this case both hidden nodes are in the same section of N_BLOCKS hidden nodes, just ignore this weight and add another one
				i--;
				continue;
			}
			Weight w(nodes + start, start);
			weight_lists[end - n_inp].push_back(w);
		}
		// weights connecting to loss node
		for (int i = 0; i < n_out; ++i) {
			weight_lists[n_tot - 1 - n_inp].push_back(Weight(nodes + n_inp + n_hid + i, n_inp + n_hid + i));
		}
		// initialize weights (he initialization, should initialize according to activation function but so far only used ReLU so should be fine),
		// not for weights connecting to loss node (wouldn't make a difference though):
		std::default_random_engine gen;
  		std::normal_distribution<double> dist(0, 1);
  		for (int i = n_inp; i < n_tot - 1; ++i) {
  			// change for other activation functions:
  			float stddev = sqrt((activation_function_hid == FAST_TANH ? 1.0 : 2.0) / (EPSILON + weight_lists[i - n_inp].size()));
  			for (int j = 0; j < weight_lists[i - n_inp].size(); ++j) {
  				weight_lists[i - n_inp][j].wt = dist(gen) * stddev;
  			}
  		}

  		// now all weights are added to the weight lists, allocate weight arrays of all nodes and copy the weight arrays there
		for (int i = n_inp; i < n_tot; ++i) {
			int n_in = weight_lists[i - n_inp].size();
			nodes[i].n_in = n_in;
			cudaMallocManaged(&nodes[i].in, sizeof(Weight) * n_in);
			std::memcpy(nodes[i].in, weight_lists[i - n_inp].data(), sizeof(Weight) * n_in);
		}

		delete [] weight_lists;
	}

	Net(Net &n) {
		// construct net as a copy of another net
		std::memcpy(this, &n, sizeof(Net));
		cudaMallocManaged(&nodes, sizeof(Node) * n_tot);
		std::memcpy(nodes, n.nodes, sizeof(Node) * n_tot);
		for (int i = 0; i < n_tot; ++i) {
			cudaMallocManaged(&nodes[i].in, sizeof(Weight) * nodes[i].n_in);
			std::memcpy(nodes[i].in, n.nodes[i].in, sizeof(Weight) * nodes[i].n_in);
		}
	}

	~Net() {
		for (int i = 0; i < n_tot; ++i) {
			cudaFree(nodes[i].in);
		}
		cudaFree(nodes);
	}

	std::pair<int, int> random_weight_indices() const {
		// returns pair of start and end index of random weight, respecting inp_weight_perc and outp_weight_perc
		while (1) {
			int start = random_unif() < weight_inp_perc ? rand() % n_inp : rand() % n_hid + n_inp;
			int end = random_unif() < weight_outp_perc ? rand() % n_out + n_inp + n_hid : rand() % n_hid + n_inp;
			
			if (start > end) {
				std::swap(start, end);
			}
			if ((end - n_inp) / N_BLOCKS == (start - n_inp) / N_BLOCKS and end < n_inp + n_hid) {
				// in this case both hidden nodes are in the same section of N_BLOCKS hidden nodes, in this case ignore this weight
				continue;
			}
			return std::pair<int, int>(start, end);
		}
	}

	void print_connections() {
		// prints connections of net 
		for (int i = n_inp; i < n_tot; ++i) {
			std::cout << i << "::  ";
			for (int j = 0; j < nodes[i].n_in; ++j) {
				std::cout << nodes[i].in[j].in_pos << ' ';
			}
			std::cout << "\n\n";
		}
	}

	void print_num_connections() {
		// prints number of in/out connections for all nodes
		std::vector<int> out_nodes;
		for (int i = 0; i < n_tot; ++i) {
			out_nodes.push_back(0);
		}
		for (int i = n_inp; i < n_tot; ++i) {
			for (int j = 0; j < nodes[i].n_in; ++j) {
				out_nodes[nodes[i].in[j].in_pos]++;
			}
		}
		for (int i = 0; i < n_tot; ++i) {
			printf("node %5d in: %5d, out: %5d\n", i, nodes[i].n_in, out_nodes[i]);
		}
	}

};

