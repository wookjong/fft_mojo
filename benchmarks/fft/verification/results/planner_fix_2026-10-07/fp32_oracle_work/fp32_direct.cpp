
#include <cmath>
#include <cstdint>
#include <iostream>
using pfUINT = uint64_t;
using pfINT = int64_t;
using VkFFTResult = int;
#define VKFFT_SUCCESS 0
#define VKFFT_BACKEND 0
#define pfceil std::ceil
struct Configuration {
    bool quadDoubleDoublePrecision = false, quadDoubleDoublePrecisionDoubleMemory = false;
    pfUINT maxThreadsNum = 1024;
    pfUINT numSharedBanks = 32, vendorID = 0x10DE, aimThreads = 128;
    pfUINT maxComputeWorkGroupSize[3] = {1024, 1024, 1024};
    bool performZeropadding[3] = {false, false, false};
};
struct VkFFTApplication { Configuration configuration; };
struct VkFFTRaderContainer {
    int registers_per_thread_per_radix[33] = {}, loc_multipliers[33] = {};
    int numSubPrimes = 0, type = 0, prime = 0, numStages = 0;
    int stageRadix[40] = {}, stage_rader_generator[40] = {};
    VkFFTRaderContainer* container = nullptr;
};
struct Integer { struct { pfUINT i = 0; } data; };
struct Spec {
    Integer fftDim;
    int numRaderPrimes = 0; VkFFTRaderContainer* raderContainer = nullptr;
    pfUINT complexSize = 8, registerBoost = 1;
    int mergeSequencesR2C = 0, useBluesteinFFT = 0, axisSwapped = 0;
    int numStages = 0, stageRadix[40] = {}, rader_generator[40] = {}, useRader = 0;
    int registers_per_thread_per_radix[33] = {};
    bool reorderFourStep = true;
};
struct Axis { Spec specializationConstants; pfUINT axisBlock[4] = {}, groupedBatch = 0; };
struct Plan { int numAxisUploads[1] = {1}; pfUINT actualFFTSizePerAxis[1][2] = {}; };
static inline VkFFTResult VkFFTGetRegistersPerThreadQuad(VkFFTApplication* app, int fft_length, int extraSharedMemoryForPow2, pfUINT max_rhs, int useRader, int* loc_multipliers, int* registers_per_thread_per_radix, int* registers_per_thread, int* min_registers_per_thread, int* isGoodSequence) {
	for (int i = 0; i < 33; i++) {
		registers_per_thread_per_radix[i] = 0;
	}
	registers_per_thread[0] = 0;
	min_registers_per_thread[0] = 10000000;

	if (loc_multipliers[2] > 0) {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					switch (loc_multipliers[2]) {
					case 1:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					case 2:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					default:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 6;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					}
				}
				else
				{
					registers_per_thread_per_radix[2] = 6;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					switch (loc_multipliers[2]) {
					case 1:
						registers_per_thread_per_radix[2] = 6;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					case 2:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					default:
						registers_per_thread_per_radix[2] = 8;
						registers_per_thread_per_radix[3] = 0;
						registers_per_thread_per_radix[5] = 5;
						registers_per_thread_per_radix[7] = 7;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
						break;
					}
				}
				else
				{
					registers_per_thread_per_radix[2] = 4;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 8;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					int max_loc_multipliers_pow2 = 0;
					pfUINT active_threads_y = max_rhs / 64; //estimate workbalance across CU (assume we have 64 CU)
					if (active_threads_y == 0) active_threads_y = 1;
					int testMinStages = 10000000;
					int maxRadixMinStages = 1;
					int fixMaxCheckRadix2 = 3;

					for (int i = 1; i <= fixMaxCheckRadix2; i++) {
						int numStages = (int)pfceil(log2(fft_length) / ((double)i));
						if (numStages < testMinStages) {
							testMinStages = numStages;
							maxRadixMinStages = i;
						}
					}
					for (int i = maxRadixMinStages; i >= 1; i--) {
						pfUINT active_threads_x = (active_threads_y * fft_length) / ((int)pow(2, i));
						if (active_threads_x >= 128) {
							max_loc_multipliers_pow2 = i;
							i = 1;
						}

					}
					if (max_loc_multipliers_pow2 < 3) max_loc_multipliers_pow2 = 3;

					int final_loc_multipliers_pow2 = 1;
					int num_stages_min = (int)log2(fft_length);
					for (int i = 2; i <= max_loc_multipliers_pow2; i++) {
						int num_stages = (int)pfceil(((int)log2(fft_length)) / (double)i);
						if (num_stages < num_stages_min) {
							final_loc_multipliers_pow2 = i;
							num_stages_min = num_stages;
						}

					}
					registers_per_thread_per_radix[2] = (loc_multipliers[2] > final_loc_multipliers_pow2) ? (int)pow(2, final_loc_multipliers_pow2) : (int)pow(2, loc_multipliers[2]);
					registers_per_thread_per_radix[2] = (loc_multipliers[2] < 3) ? (int)pow(2, loc_multipliers[2]) : registers_per_thread_per_radix[2];
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
		}
	}
	else {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 3;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 6;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					if (loc_multipliers[3] == 1) {
						registers_per_thread_per_radix[2] = 0;
						registers_per_thread_per_radix[3] = 3;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 0;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
					}
					else {
						registers_per_thread_per_radix[2] = 0;
						registers_per_thread_per_radix[3] = 9;
						registers_per_thread_per_radix[5] = 0;
						registers_per_thread_per_radix[7] = 0;
						registers_per_thread_per_radix[11] = 0;
						registers_per_thread_per_radix[13] = 0;
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 5;
					registers_per_thread_per_radix[7] = 0;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					registers_per_thread_per_radix[2] = 0;
					registers_per_thread_per_radix[3] = 0;
					registers_per_thread_per_radix[5] = 0;
					registers_per_thread_per_radix[7] = 7;
					registers_per_thread_per_radix[11] = 0;
					registers_per_thread_per_radix[13] = 0;
				}
				else
				{
					min_registers_per_thread[0] = 2;
					registers_per_thread[0] = 2;
					//Rader-only sequence
					//return VKFFT_ERROR_UNSUPPORTED_RADIX;
				}
			}
		}

	}

	registers_per_thread_per_radix[32] = ((registers_per_thread_per_radix[2] % 32) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[16] = ((registers_per_thread_per_radix[2] % 16) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[8] = ((registers_per_thread_per_radix[2] % 8) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[4] = ((registers_per_thread_per_radix[2] % 4) == 0) ? registers_per_thread_per_radix[2] : 0;
	if ((registers_per_thread_per_radix[2] >= 12) && (registers_per_thread_per_radix[3] >= 12)) {
		registers_per_thread_per_radix[12] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
		if ((registers_per_thread_per_radix[12] % 12) != 0) registers_per_thread_per_radix[12] = 0;
	}
	registers_per_thread_per_radix[6] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[9] = ((registers_per_thread_per_radix[3] % 9) == 0) ? registers_per_thread_per_radix[3] : 0;
	registers_per_thread_per_radix[10] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[14] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[7]) ? registers_per_thread_per_radix[7] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[15] = (registers_per_thread_per_radix[3] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[3];

	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}
	if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0])) isGoodSequence[0] = 0;
	else isGoodSequence[0] = 1;
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTGetRegistersPerThread(VkFFTApplication* app, int fft_length, int extraSharedMemoryForPow2, pfUINT max_rhs, int useRader, int* loc_multipliers, int* registers_per_thread_per_radix, int* registers_per_thread, int* min_registers_per_thread, int* isGoodSequence) {
	if (app->configuration.quadDoubleDoublePrecision || app->configuration.quadDoubleDoublePrecisionDoubleMemory) {
		VkFFTGetRegistersPerThreadQuad(app, fft_length, extraSharedMemoryForPow2, max_rhs, useRader, loc_multipliers, registers_per_thread_per_radix, registers_per_thread, min_registers_per_thread, isGoodSequence);
		return VKFFT_SUCCESS;
	}
	for (int i = 0; i < 33; i++) {
		registers_per_thread_per_radix[i] = 0;
	}
	registers_per_thread[0] = 0;
	min_registers_per_thread[0] = 10000000;

	if (loc_multipliers[2] > 0) {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;

						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 15;

								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 15;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								break;
							}
							registers_per_thread_per_radix[5] = 10;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 5;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 10;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 10;
								break;
							}
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;

						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 12;
							registers_per_thread_per_radix[3] = 12;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							if (loc_multipliers[2] == loc_multipliers[3]) {
								registers_per_thread_per_radix[2] = 6;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
							else {
								switch (loc_multipliers[2]) {
								case 1:
									registers_per_thread_per_radix[2] = 6;
									registers_per_thread_per_radix[3] = 6;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								case 2:
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								default:
									registers_per_thread_per_radix[2] = 12;
									registers_per_thread_per_radix[3] = 12;
									registers_per_thread_per_radix[5] = 0;
									registers_per_thread_per_radix[7] = 0;
									registers_per_thread_per_radix[11] = 0;
									registers_per_thread_per_radix[13] = 0;
									break;
								}
							}
						}
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 10;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 3:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 16;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 14;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 10;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
								break;
							}
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							switch (loc_multipliers[2]) {
							case 1:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							case 2:
								registers_per_thread_per_radix[2] = 12;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							default:
								registers_per_thread_per_radix[2] = 8;
								registers_per_thread_per_radix[3] = 0;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
								break;
							}
						}
						else {
							int max_loc_multipliers_pow2 = 0;
							pfUINT active_threads_y = max_rhs / 64; //estimate workbalance across CU (assume we have 64 CU)
							if (active_threads_y == 0) active_threads_y = 1;
							int testMinStages = 10000000;
							int maxRadixMinStages = 1;
							int fixMaxCheckRadix2 = 3;
#if(VKFFT_BACKEND==1)
							fixMaxCheckRadix2 = (((fft_length >= 1024) || (fft_length == 256)) && (extraSharedMemoryForPow2) && (!useRader)) ? 5 : 3;
#endif
							for (int i = 1; i <= fixMaxCheckRadix2; i++) {
								int numStages = (int)pfceil(log2(fft_length) / ((double)i));
								if (numStages < testMinStages) {
									testMinStages = numStages;
									maxRadixMinStages = i;
								}
							}
							for (int i = maxRadixMinStages; i >= 1; i--) {
								pfUINT active_threads_x = (active_threads_y * fft_length) / ((int)pow(2, i));
								if (active_threads_x >= 128) {
									max_loc_multipliers_pow2 = i;
									i = 1;
								}

							}
							if (max_loc_multipliers_pow2 < 3) max_loc_multipliers_pow2 = 3;

							int final_loc_multipliers_pow2 = 1;
							int num_stages_min = (int)log2(fft_length);
							for (int i = 2; i <= max_loc_multipliers_pow2; i++) {
								int num_stages = (int)pfceil(((int)log2(fft_length)) / (double)i);
								if (num_stages < num_stages_min) {
									final_loc_multipliers_pow2 = i;
									num_stages_min = num_stages;
								}

							}
							registers_per_thread_per_radix[2] = (loc_multipliers[2] > final_loc_multipliers_pow2) ? (int)pow(2, final_loc_multipliers_pow2) : (int)pow(2, loc_multipliers[2]);
							registers_per_thread_per_radix[2] = (loc_multipliers[2] < 3) ? (int)pow(2, loc_multipliers[2]) : registers_per_thread_per_radix[2];
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
		}
	}
	else {
		if (loc_multipliers[3] > 0) {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 14;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 15;
							registers_per_thread_per_radix[5] = 15;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[3] == 1) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 14;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 6;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 7;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
				}
				else {
					if (loc_multipliers[3] == 1) {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 12;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 3;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
					else {
						if (loc_multipliers[11] > 0) {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 11;
								registers_per_thread_per_radix[13] = 0;
							}
						}
						else {
							if (loc_multipliers[13] > 0) {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 13;
							}
							else {
								registers_per_thread_per_radix[2] = 0;
								registers_per_thread_per_radix[3] = 9;
								registers_per_thread_per_radix[5] = 0;
								registers_per_thread_per_radix[7] = 0;
								registers_per_thread_per_radix[11] = 0;
								registers_per_thread_per_radix[13] = 0;
							}
						}
					}
				}
			}
		}
		else {
			if (loc_multipliers[5] > 0) {
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 5;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
			}
			else
			{
				if (loc_multipliers[7] > 0) {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 7;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 0;
						}
					}
				}
				else {
					if (loc_multipliers[11] > 0) {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 11;
							registers_per_thread_per_radix[13] = 0;
						}
					}
					else {
						if (loc_multipliers[13] > 0) {
							registers_per_thread_per_radix[2] = 0;
							registers_per_thread_per_radix[3] = 0;
							registers_per_thread_per_radix[5] = 0;
							registers_per_thread_per_radix[7] = 0;
							registers_per_thread_per_radix[11] = 0;
							registers_per_thread_per_radix[13] = 13;
						}
						else {
							min_registers_per_thread[0] = 2;
							registers_per_thread[0] = 2;
							//Rader-only sequence
							//return VKFFT_ERROR_UNSUPPORTED_RADIX;
						}
					}
				}
			}
		}

	}

	registers_per_thread_per_radix[32] = ((registers_per_thread_per_radix[2] % 32) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[16] = ((registers_per_thread_per_radix[2] % 16) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[8] = ((registers_per_thread_per_radix[2] % 8) == 0) ? registers_per_thread_per_radix[2] : 0;
	registers_per_thread_per_radix[4] = ((registers_per_thread_per_radix[2] % 4) == 0) ? registers_per_thread_per_radix[2] : 0;
	if ((registers_per_thread_per_radix[2] >= 12) && (registers_per_thread_per_radix[3] >= 12)) {
		registers_per_thread_per_radix[12] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
		if ((registers_per_thread_per_radix[12] % 12) != 0) registers_per_thread_per_radix[12] = 0;
	}
	registers_per_thread_per_radix[6] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[3]) ? registers_per_thread_per_radix[3] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[9] = ((registers_per_thread_per_radix[3] % 9) == 0) ? registers_per_thread_per_radix[3] : 0;
	registers_per_thread_per_radix[10] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[14] = (registers_per_thread_per_radix[2] > registers_per_thread_per_radix[7]) ? registers_per_thread_per_radix[7] : registers_per_thread_per_radix[2];
	registers_per_thread_per_radix[15] = (registers_per_thread_per_radix[3] > registers_per_thread_per_radix[5]) ? registers_per_thread_per_radix[5] : registers_per_thread_per_radix[3];

	for (int i = 0; i < 33; i++) {
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread[0])) min_registers_per_thread[0] = registers_per_thread_per_radix[i];
		if ((registers_per_thread_per_radix[i] != 0) && (registers_per_thread_per_radix[i] > registers_per_thread[0])) registers_per_thread[0] = registers_per_thread_per_radix[i];
	}
	if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0])) isGoodSequence[0] = 0;
	else isGoodSequence[0] = 1;
	return VKFFT_SUCCESS;
}
static inline VkFFTResult VkFFTOptimizeRadixKernels(int* registers_per_thread_per_radix, int* loc_multipliers, int registerBoost, int* maxNonPow2Radix, int* reqLocRegs, VkFFTRaderContainer* raderContainer, int numRaderPrimes) {
	VkFFTResult res = VKFFT_SUCCESS;
	if (numRaderPrimes) {
		for (int i = 0; i < numRaderPrimes; i++) {
			res = VkFFTOptimizeRadixKernels(raderContainer[i].registers_per_thread_per_radix, raderContainer[i].loc_multipliers, 1, maxNonPow2Radix, reqLocRegs, raderContainer[i].container, raderContainer[i].numSubPrimes);
			if (res != VKFFT_SUCCESS) return res;
		}
	}
	//optimize used radix kernels
	if (((registers_per_thread_per_radix[32] > 0) || ((registers_per_thread_per_radix[2] % 32) == 0)) && ((registers_per_thread_per_radix[32]) % 32 == 0) && (loc_multipliers[2] >= 5)) {
		loc_multipliers[32] = loc_multipliers[2] / 5;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[32] * 5;
		if ((registers_per_thread_per_radix[2] % 32) == 0) registers_per_thread_per_radix[32] = registers_per_thread_per_radix[2];
	}
	if (((registers_per_thread_per_radix[16] > 0) || ((registers_per_thread_per_radix[2] % 16) == 0)) && ((registers_per_thread_per_radix[16]) % 16 == 0) && (loc_multipliers[2] >= 4)) {
		loc_multipliers[16] = loc_multipliers[2] / 4;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[16] * 4;
		if ((registers_per_thread_per_radix[2] % 16) == 0) registers_per_thread_per_radix[16] = registers_per_thread_per_radix[2];//if we got 16 regs, why not use r16 kernel
	}
	if ((registers_per_thread_per_radix[15] > 0) && ((registers_per_thread_per_radix[15]) % 15 == 0) && (loc_multipliers[3] >= 1) && (loc_multipliers[5] >= 1)) {
		loc_multipliers[15] = (loc_multipliers[3] > loc_multipliers[5]) ? loc_multipliers[5] : loc_multipliers[3];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[15];
		loc_multipliers[5] = loc_multipliers[5] - loc_multipliers[15];
	}
	if ((registers_per_thread_per_radix[14] > 0) && ((registers_per_thread_per_radix[14]) % 14 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[7] >= 1)) {
		loc_multipliers[14] = (loc_multipliers[2] > loc_multipliers[7]) ? loc_multipliers[7] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[14];
		loc_multipliers[7] = loc_multipliers[7] - loc_multipliers[14];
	}
	if ((registers_per_thread_per_radix[12] > 0) && ((registers_per_thread_per_radix[12]) % 12 == 0) && (loc_multipliers[2] >= 2) && (loc_multipliers[3] >= 1)) {
		loc_multipliers[12] = (loc_multipliers[2] > 2 * loc_multipliers[3]) ? loc_multipliers[3] : loc_multipliers[2] / 2;
		loc_multipliers[2] = loc_multipliers[2] - 2 * loc_multipliers[12];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[12];
	}
	if ((registers_per_thread_per_radix[10] > 0) && ((registers_per_thread_per_radix[10]) % 10 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[5] >= 1)) {
		loc_multipliers[10] = (loc_multipliers[2] > loc_multipliers[5]) ? loc_multipliers[5] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[10];
		loc_multipliers[5] = loc_multipliers[5] - loc_multipliers[10];
	}
	if ((registers_per_thread_per_radix[9] > 0) && ((registers_per_thread_per_radix[9]) % 9 == 0) && (loc_multipliers[3] >= 2)) {
		loc_multipliers[9] = loc_multipliers[3] / 2;
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[9] * 2;
	}
	if (((registers_per_thread_per_radix[8] > 0) || ((registers_per_thread_per_radix[2] % 8) == 0)) && ((registers_per_thread_per_radix[8]) % 8 == 0) && (loc_multipliers[2] >= 3)) {
		loc_multipliers[8] = loc_multipliers[2] / 3;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[8] * 3;
		if ((registers_per_thread_per_radix[2] % 8) == 0) registers_per_thread_per_radix[8] = registers_per_thread_per_radix[2];
	}
	if ((registers_per_thread_per_radix[6] > 0) && ((registers_per_thread_per_radix[6]) % 6 == 0) && (loc_multipliers[2] >= 1) && (loc_multipliers[3] >= 1)) {
		loc_multipliers[6] = (loc_multipliers[2] > loc_multipliers[3]) ? loc_multipliers[3] : loc_multipliers[2];
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[6];
		loc_multipliers[3] = loc_multipliers[3] - loc_multipliers[6];
	}
	if (((registers_per_thread_per_radix[4] > 0) || ((registers_per_thread_per_radix[2] % 4) == 0)) && ((registers_per_thread_per_radix[4]) % 4 == 0) && (loc_multipliers[2] >= 2)) {
		loc_multipliers[4] = loc_multipliers[2] / 2;
		loc_multipliers[2] = loc_multipliers[2] - loc_multipliers[4] * 2;
		if ((registers_per_thread_per_radix[2] % 4) == 0) registers_per_thread_per_radix[4] = registers_per_thread_per_radix[2];
	}
	if ((registerBoost == 2) && (loc_multipliers[2] == 0)) {
		if (loc_multipliers[4] > 0) {
			loc_multipliers[4]--;
			loc_multipliers[2] = 2;
		}
		else if (loc_multipliers[8] > 0) {
			loc_multipliers[8]--;
			loc_multipliers[4]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[16] > 0) {
			loc_multipliers[16]--;
			loc_multipliers[8]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[32] > 0) {
			loc_multipliers[32]--;
			loc_multipliers[16]++;
			loc_multipliers[2]++;
		}
	}
	if ((registerBoost == 4) && (loc_multipliers[4] == 0)) {
		if (loc_multipliers[8] > 0) {
			loc_multipliers[8]--;
			loc_multipliers[4]++;
			loc_multipliers[2]++;
		}
		else if (loc_multipliers[16] > 0) {
			if (loc_multipliers[2] == 0) {
				loc_multipliers[16]--;
				loc_multipliers[4] = 2;
			}
			else {
				loc_multipliers[16]--;
				loc_multipliers[4]++;
				loc_multipliers[2]--;
				loc_multipliers[8]++;
			}
		}
		else if (loc_multipliers[32] > 0) {
			if (loc_multipliers[2] == 0) {
				loc_multipliers[32]--;
				loc_multipliers[8]++;
				loc_multipliers[4]++;
			}
			else {
				loc_multipliers[32]--;
				loc_multipliers[16]++;
				loc_multipliers[4]++;
				loc_multipliers[2]--;
			}
		}
	}
	for (int i = 2; i < 33; i++) {
		int usedLocRegs = 0;
		if (loc_multipliers[i] > 0) {
			switch (i) {
			case 6:
				usedLocRegs = 3;
				break;
			case 9:
				usedLocRegs = 3;
				break;
			case 10:
				usedLocRegs = 5;
				break;
			case 12:
				usedLocRegs = 3;
				break;
			case 14:
				usedLocRegs = 7;
				break;
			case 15:
				usedLocRegs = 5;
				break;
			default:
				usedLocRegs = i;
				break;
			}
		}
		if ((loc_multipliers[i] > 0) && ((i & (i - 1)) != 0) && (i > maxNonPow2Radix[0])) {
			maxNonPow2Radix[0] = i;
		}
		if ((usedLocRegs > reqLocRegs[0]) && ((i & (i - 1)) != 0)) {
			reqLocRegs[0] = usedLocRegs;
		}
	}
	return res;
}
static inline VkFFTResult VkFFTMinMaxRegisterCheck(int numStages, int* stageRadix, int* min_registers_per_thread, int* registers_per_thread, int* registers_per_thread_per_radix, VkFFTRaderContainer* raderContainer, int numRaderPrimes, int* stage_rader_generator) {
	VkFFTResult res = VKFFT_SUCCESS;
	for (pfINT j = 0; j < (pfINT)numStages; j++) {
		if (stage_rader_generator[j] == 0) {
			if (registers_per_thread_per_radix[stageRadix[j]] > 0) {
				if (registers_per_thread_per_radix[stageRadix[j]] < min_registers_per_thread[0]) {
					min_registers_per_thread[0] = registers_per_thread_per_radix[stageRadix[j]];
				}
				if (registers_per_thread_per_radix[stageRadix[j]] > registers_per_thread[0]) {
					registers_per_thread[0] = registers_per_thread_per_radix[stageRadix[j]];
				}
			}
		}
		else {
			for (pfINT i = 0; i < (pfINT)numRaderPrimes; i++) {
				if (raderContainer[i].prime == stageRadix[j]) {
					if (raderContainer[i].type == 0) {
						for (pfINT j2 = 0; j2 < (pfINT)raderContainer[i].numStages; j2++) {
							if (raderContainer[i].stage_rader_generator[j] == 0) {
								if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] > 0) {
									if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] < min_registers_per_thread[0]) {
										min_registers_per_thread[0] = raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]];
									}
									if (raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]] > registers_per_thread[0]) {
										registers_per_thread[0] = raderContainer[i].registers_per_thread_per_radix[raderContainer[i].stageRadix[j2]];
									}
								}
							}
							else {
								res = VkFFTMinMaxRegisterCheck(raderContainer[i].numStages, raderContainer[i].stageRadix, min_registers_per_thread, registers_per_thread, raderContainer[i].registers_per_thread_per_radix, raderContainer[i].container, raderContainer[i].numSubPrimes, raderContainer[i].stage_rader_generator);
								if (res != VKFFT_SUCCESS) return res;
							}
						}
					}
				}
			}
		}
	}
	return res;
}

int main() {
    char mode; int n, batch, passes, k;
    while (std::cin >> mode >> n >> batch >> passes >> k) {
        VkFFTApplication application; auto* app = &application;
        int loc_multipliers[33] = {}, registers_per_thread_per_radix[33] = {};
        int registers_per_thread = 0, min_registers_per_thread = 10000000, good = 0;
        int remaining = n;
        for (int p: {2,3,5,7}) while (remaining % p == 0) { remaining /= p; ++loc_multipliers[p]; }
        if (remaining != 1) return 2;
        VkFFTGetRegistersPerThread(app, n, 1, batch, 0, loc_multipliers,
            registers_per_thread_per_radix, &registers_per_thread, &min_registers_per_thread, &good);
        if (mode == 'R') {
            for (int p: {2,3,5,7}) std::cout << registers_per_thread_per_radix[p] << " ";
            std::cout << "\n"; continue;
        }
        Spec sc;
        pfUINT axis_id = 0, maxBatchCoalesced = passes == 1 ? 1 : 4;
        pfUINT locAxisSplit[3] = {pfUINT(n), pfUINT(n), pfUINT(n)};
        pfUINT registerBoost = 1, estimate_rader_threadnum = 0, temp = 0;
		if ((int)pfceil((maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost)) > app->configuration.maxThreadsNum) || (sc.useRader && (estimate_rader_threadnum > app->configuration.maxThreadsNum)))
		{
			int scaleRegistersNum = 1;
			if ((axis_id == 0) && (k == 0) && (maxBatchCoalesced > 1)) {
				maxBatchCoalesced = (int)(app->configuration.maxThreadsNum * (min_registers_per_thread * registerBoost) / locAxisSplit[k]);
				if (maxBatchCoalesced < 1) maxBatchCoalesced = 1;
			}
			if (((int)pfceil(maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost * scaleRegistersNum))) > app->configuration.maxThreadsNum) {
				for (int i = 2; i < locAxisSplit[k]; i++) {
					if ((((int)pfceil(maxBatchCoalesced * locAxisSplit[k] / (double)(min_registers_per_thread * registerBoost * i))) <= app->configuration.maxThreadsNum)) {
						scaleRegistersNum = i;
						i = (int)locAxisSplit[k];
					}
				}
			}
			min_registers_per_thread *= scaleRegistersNum;
			registers_per_thread *= scaleRegistersNum;
			for (int i = 2; i < 33; i++) {
				if (registers_per_thread_per_radix[i] != 0) {
					registers_per_thread_per_radix[i] *= scaleRegistersNum;
				}
			}
			int new_min_registers = 10000000;
			for (int i = 2; i < 33; i++) {
				if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < new_min_registers)) new_min_registers = registers_per_thread_per_radix[i];
			}
			for (pfINT i = 0; i < (pfINT)sc.numRaderPrimes; i++) {
				if (sc.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if ((sc.raderContainer[i].registers_per_thread_per_radix[j] > 0) && (sc.raderContainer[i].registers_per_thread_per_radix[j] < new_min_registers)) new_min_registers = sc.raderContainer[i].registers_per_thread_per_radix[j];
					}
				}
			}
			if ((int)pfceil((maxBatchCoalesced * locAxisSplit[k] / (double)(new_min_registers * registerBoost))) > app->configuration.maxThreadsNum) {
				// if we get here, there can be trouble with small primes, as we can have one thread do at max one fftDim. This is only an issue for small primes in sequences close to shared memory limit sizes for extremely big shared memory sizes (>136KB)
				for (pfINT i = 0; i < (pfINT)sc.numRaderPrimes; i++) {
					if (sc.raderContainer[i].type == 0) {
						for (int j = 2; j < 33; j++) {
							if (sc.raderContainer[i].registers_per_thread_per_radix[j] != 0) {
								sc.raderContainer[i].registers_per_thread_per_radix[j] *= scaleRegistersNum;
							}
						}
					}
				}
			}
			else {
				min_registers_per_thread = new_min_registers;
			}
			if (min_registers_per_thread > registers_per_thread) {
				temp = min_registers_per_thread;
				min_registers_per_thread = registers_per_thread;
				registers_per_thread = (int)temp;
			}
			for (int i = 2; i < 33; i++) {
				if (registers_per_thread_per_radix[i] > registers_per_thread) {
					registers_per_thread = registers_per_thread_per_radix[i];
				}
				if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread)) {
					min_registers_per_thread = registers_per_thread_per_radix[i];
				}
			}
			for (pfINT i = 0; i < (pfINT)sc.numRaderPrimes; i++) {
				if (sc.raderContainer[i].type == 0) {
					for (int j = 2; j < 33; j++) {
						if (sc.raderContainer[i].registers_per_thread_per_radix[j] > registers_per_thread) {
							registers_per_thread = sc.raderContainer[i].registers_per_thread_per_radix[j];
						}
						if ((sc.raderContainer[i].registers_per_thread_per_radix[j] > 0) && (sc.raderContainer[i].registers_per_thread_per_radix[j] < min_registers_per_thread)) {
							min_registers_per_thread = sc.raderContainer[i].registers_per_thread_per_radix[j];
						}
					}
				}
			}
			if ((loc_multipliers[3] >= 2) && (((registers_per_thread / min_registers_per_thread) % 3) == 0) && (sc.numRaderPrimes == 0)) {
				registers_per_thread /= 3;
				for (int i = 2; i < 33; i++) {
					if (registers_per_thread_per_radix[i] % 9 == 0) {
						registers_per_thread_per_radix[i] /= 3;
					}
				}
				for (int i = 2; i < 33; i++) {
					if (registers_per_thread_per_radix[i] > registers_per_thread) {
						registers_per_thread = registers_per_thread_per_radix[i];
					}
					if ((registers_per_thread_per_radix[i] > 0) && (registers_per_thread_per_radix[i] < min_registers_per_thread)) {
						min_registers_per_thread = registers_per_thread_per_radix[i];
					}
				}
			}
		}
        int maxradix = 1, usedregs = 1;
        VkFFTOptimizeRadixKernels(registers_per_thread_per_radix, loc_multipliers,
                                 1, &maxradix, &usedregs, nullptr, 0);
        for (int i=2; i<33; ++i) sc.registers_per_thread_per_radix[i] = registers_per_thread_per_radix[i];
        for (int i=32; i>1; --i) while (loc_multipliers[i]>0) {
            sc.stageRadix[sc.numStages++] = i; --loc_multipliers[i];
        }
        min_registers_per_thread = 10000000; registers_per_thread = 0;
        VkFFTMinMaxRegisterCheck(sc.numStages, sc.stageRadix, &min_registers_per_thread,
            &registers_per_thread, sc.registers_per_thread_per_radix, nullptr, 0, sc.rader_generator);
        int switchRegisterBoost = 0;
        if (switchRegisterBoost > 0) {}
		else {
			//try to read directly to registers
			if (min_registers_per_thread != registers_per_thread) {
				for (int i = 0; i < sc.numStages; i++) {
					if (sc.registers_per_thread_per_radix[sc.stageRadix[i]] == min_registers_per_thread) {
						int stageid = sc.stageRadix[i];
						sc.stageRadix[i] = sc.stageRadix[0];
						sc.stageRadix[0] = stageid;
						if (sc.useRader) {
							stageid = sc.rader_generator[i];
							sc.rader_generator[i] = sc.rader_generator[0];
							sc.rader_generator[0] = stageid;
						}
						i = sc.numStages;
					}
				}
			}
		}
        std::cout << min_registers_per_thread << " " << registers_per_thread << " " << sc.numStages;
        for (int i=0; i<sc.numStages; ++i) std::cout << " " << sc.stageRadix[i];
        for (int i=2; i<33; ++i) std::cout << " " << sc.registers_per_thread_per_radix[i];
        std::cout << "\n";
    }
}
