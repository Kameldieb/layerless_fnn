// class for loading samples from a dataset and applying augmentation, runs on cpu (for now)
// works for mnist (including loading the dataset and applying elastic deformations for augmentation), didn't try other datasets

// todo: use const everywhere possible so dataset can't be changed by the "user"


#include "mnist/include/mnist/mnist_reader.hpp"

#define MNIST_DATA_LOCATION "mnist"

// samples consist of a pair of two float arrays, input and output
typedef std::pair<float*, float*> sample;


float random_unif(float a=0, float b=1) {
	return a + (b-a) * rand()/RAND_MAX;
}

class SampleLoader {

private:
	sample *training_samples;
	sample *test_samples;
	// type of augmentation, only implemented elastic deformations for image datasets so far
	std::string augmentation = "none";
	float max_shift_x;
	float max_shift_y;
	float *augmented_input;

	float *augment(float *a) {
		if (augmentation == "elastic") {
			// uses polynomial elastic deformations for augmentation (no idea if this is how it's usually done)
			// basically applies shifts in x and y direction that are second degree polynomials in x and y with random coefficients 

			float x_params[6];
			float y_params[6];
			for (int i = 0; i < 6; ++i) {
				x_params[i] = random_unif(-1, 1);
				y_params[i] = random_unif(-1, 1);
			}
			// now scale numbers ensuring that maximum x shift is max_shift_x, same with y
			float hdx = 2.0/width;
			float hdy = 2.0/height;
			x_params[0] *= max_shift_x / 6;
			x_params[1] *= max_shift_x / 6 * hdx;
			x_params[2] *= max_shift_x / 6 * hdx;
			x_params[3] *= max_shift_x / 6 * hdx * hdy;
			x_params[4] *= max_shift_x / 6 * hdx * hdx;
			x_params[5] *= max_shift_x / 6 * hdy * hdy;
			y_params[0] *= max_shift_y / 6;
			y_params[1] *= max_shift_y / 6 * hdx;
			y_params[2] *= max_shift_y / 6 * hdx;
			y_params[3] *= max_shift_y / 6 * hdx * hdy;
			y_params[4] *= max_shift_y / 6 * hdx * hdx;
			y_params[5] *= max_shift_y / 6 * hdy * hdy;

			for (int i = 0; i < width*height; ++i) {
				int x = i % height;
				int y = i / height;
				// center
				float xf = (float)x - width/2.0;
				float yf = (float)y - height/2.0;

				// calculate values to take pixels from, applying transformations
				float temp_shift_x = x_params[0] + x_params[1] * xf + x_params[2] * yf + x_params[3] * xf * yf + x_params[4] * xf * xf + x_params[5] * yf * yf;
				yf += y_params[0] + y_params[1] * xf + y_params[2] * yf + y_params[3] * xf * yf + y_params[4] * xf * xf + y_params[5] * yf * yf;
				xf += temp_shift_x;

				// // remove centering
				xf += width/2.0;
				yf += height/2.0;

				// round to nearest integer
				int newx = (int)(xf + 0.5);
				int newy = (int)(yf + 0.5);

				if (newx >= 0 && newx < width && newy >= 0 && newy < height) {
					// new value taken from picture
					int new_index = newx + newy * height;
					augmented_input[i] = a[new_index];
					if (color) {
						augmented_input[i + width*height] = a[new_index + width*height];
						augmented_input[i + 2*width*height] = a[new_index + 2*width*height];
					}
				}
				else {
					// padding with 0
					augmented_input[i] = 0;
					if (color) {
						augmented_input[i + width*height] = 0;
						augmented_input[i + 2*width*height] = 0;
					}
				}
			}
		}
		else if (augmentation == "none") {
			// could copy a to augmented_input instead so the original data can't be changed, but should use const everywhere instead anyway...
			augmented_input = a;
		}
		else {
			throw std::invalid_argument("augmentation type unknown");
		}
		
		return augmented_input;
	}

public:
	// input and output sizes
	int in_size;
	int out_size;
	// number of training and test samples
	int n_training;
	int n_test;
	// used for augmentation if inputs are images:
	int width;
	int height;
	bool color;


	void load_mnist() {
	    // Load MNIST data, images as floats in range [0, 1], labels as one-hot-vectors
	    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
	        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
	    in_size = dataset.training_images[0].size();	// should be 784
	    out_size = 10;
	    n_training = dataset.training_images.size();
	    n_test = dataset.test_images.size();
	    width = 28;
	    height = 28;
	    assert(in_size == 784 && n_training == 60000 && n_test == 10000);	// check if dataset is correct
	    // convert dataset from vectors to samples
	    cudaMallocManaged(&training_samples, sizeof(sample) * n_training);
	    cudaMallocManaged(&test_samples, sizeof(sample) * n_test);
		for (int i = 0; i < n_training; ++i) {
			cudaMallocManaged(&training_samples[i].first, sizeof(float) * in_size);
			cudaMallocManaged(&training_samples[i].second, sizeof(float) * out_size);
			for (int j = 0; j < out_size; ++j) {
				training_samples[i].second[j] = 0.0;
			}
	    	training_samples[i].second[dataset.training_labels[i]] = 1.0;
			for (int j = 0; j < in_size; ++j) {
				training_samples[i].first[j] = dataset.training_images[i][j]/256.0;
	    	}
	    }
	    for (int i = 0; i < n_test; ++i) {
			cudaMallocManaged(&test_samples[i].first, sizeof(float) * in_size);
			cudaMallocManaged(&test_samples[i].second, sizeof(float) * out_size);
			for (int j = 0; j < out_size; ++j) {
				test_samples[i].second[j] = 0.0;
			}
	    	test_samples[i].second[dataset.test_labels[i]] = 1.0;
			for (int j = 0; j < in_size; ++j) {
				test_samples[i].first[j] = dataset.test_images[i][j]/256.0;
	    	}
	    }
	    color=false;
	    cudaMallocManaged(&augmented_input, sizeof(float) * in_size);
	}

	void set_elastic_augmentation(float x_shift, float y_shift) {
		// used to activate elastic augmentation for image inputs
		augmentation = "elastic";
		max_shift_x = x_shift;
		max_shift_y = y_shift;
	}

	sample load_training_sample() {
		// returns dynamically allocated array with augmented training-sample
		int index = rand() % n_training;
		sample orig = training_samples[index];
		sample augmented;
		augmented.first = augment(orig.first);
		augmented.second = orig.second;
		return augmented;
	}
	
	sample *load_training_samples(int n) {
		// return array of n (augmented) trainings samples
		sample *res = new sample[n];
		for (int i = 0; i < n; ++i) {
			res[i] = load_training_sample();
		}
		return res;
	}

	sample *load_test_samples() {
		// simply returns the test samples
		return test_samples;
	}
};