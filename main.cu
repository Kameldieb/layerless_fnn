
#include <iostream>
#include <stdio.h>
#include <string>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <assert.h>
#include <chrono>
#include <algorithm>


// number of threads and blocks, no idea how to ideally choose these
// during forward and backward pass, N_BLOCKS nodes are done in parallel
#define N_BLOCKS 32
#define N_THREADS 128



#include "Weight.h"
#include "SampleLoader.h"
#include "Node.h"
#include "Net.h"
#include "Train.h"

int main(void) {

	cudaDeviceReset();

	srand(123);

	SampleLoader loader;
	loader.load_mnist();
	loader.set_elastic_augmentation(8, 8);

	Net net_host(loader.in_size, 512, loader.out_size, 1000, 0.5, 0.1, LEAKY_RELU);
	Net *net;
	cudaMallocManaged(&net, sizeof(Net));
	std::memcpy(net, &net_host, sizeof(Net));


	for (int i = 0; i < 1000; ++i) {
		if (i % 100 == 0 and i > 0) net->print_connections();
		std::cout << "training nr. " << i << std::flush;
		train_and_restructure(net, loader, i > 190 ? 10 : 20, 10000, 1, 1.0/(i+100));
		std::cout <<  " done\n";
		std::cout << "test set correct: " << test(net, loader) << '\n';
	}


	cudaDeviceReset();

	return 0;

}
