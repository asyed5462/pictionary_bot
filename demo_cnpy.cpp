#include "cnpy.h"
#include <iostream>
#include <string>
#include <assert.h>

using namespace std;

/* A simple function trying to use cnpy library */
int demo_cnpy(string input_npy) {
	cnpy::NpyArray data = cnpy::npy_load(input_npy);

	// Data type - Google Draw dataset in npy should use uint8_t
	cout << "Word size is " << data.word_size << " byte(s)." << endl;

	// Dimension
	cout << "Dimension of " << input_npy << " is: [";
	for (auto it = data.shape.begin(); it != data.shape.end(); it++) {
		if (it != data.shape.begin()) {
			cout << ", ";
		}
		cout << *it;
	}
	cout << "]" << endl;

	assert(data.shape.size() == 2); // Dimension should be n x 784
	assert(data.shape[1] == 784);

	uint8_t* data_ptr = data.data<uint8_t>();
	// Print the first few bitmaps
	cout << endl << "NOTE: no grayscale information in this terminal print!" << endl;
	constexpr int num_bitmaps_to_print = 3;
	assert(data.shape[0] > num_bitmaps_to_print);
	for (int i = 0; i < num_bitmaps_to_print; i++) {
		cout << "Bitmap " << i << ": ";
		for (int y = 0; y < 28; y++) { // 28 * 28 = 784
			for (int x = 0; x < 28; x++) {
				char c = data_ptr[i*784 + y*28 + x] > 0 ? '.' : ' ';
				cout << c;
			}
			cout << endl;
		}
		cout << endl;
	}

	return 0;
}

int main(int argc, char *argv[]) {

	if (argc != 2) {
		cout << "Incorrect command line arguments." << endl;
		cout << "Correct usage: ./demo_cnpy <input-npy-file>" << endl;
		return 1;
	}

	demo_cnpy(string(argv[1]));
	return 0;
}
