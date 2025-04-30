#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define ALPHABET_SIZE 4
#define MAX_NODES 10000
#define THREADS_PER_BLOCK 256

__device__ int d_gotoFn[MAX_NODES][ALPHABET_SIZE];
__device__ int d_output[MAX_NODES];

__host__ __device__ int dnaToIndex(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

__global__ void ahoCorasickParallelGPU(const char* text, int textLength, int* matches) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= textLength) return;

    int state = 0;
    for (int i = idx; i < textLength; ++i) {
        int c = dnaToIndex(text[i]);
        if (c == -1) break;

        state = d_gotoFn[state][c];
        if (state == -1) break;

        if (d_output[state] > 0) {
            atomicAdd(&matches[idx], d_output[state]);
        }
    }
}

void buildFSM(const std::vector<std::string>& patterns, int gotoFn[][ALPHABET_SIZE], int output[], int& statesCount) {
    memset(gotoFn[0], -1, sizeof(int) * ALPHABET_SIZE);
    output[0] = 0;
    statesCount = 1;

    for (const auto& pattern : patterns) {
        int currentState = 0;
        for (char c : pattern) {
            int idx = dnaToIndex(c);
            if (idx == -1) continue;

            if (gotoFn[currentState][idx] == -1) {
                memset(gotoFn[statesCount], -1, sizeof(int) * ALPHABET_SIZE);
                output[statesCount] = 0;
                gotoFn[currentState][idx] = statesCount++;
            }
            currentState = gotoFn[currentState][idx];
        }
        output[currentState]++;
    }
}

void loadPatterns(const char* filename, std::vector<std::string>& patterns) {
    std::ifstream file(filename);
    std::string line;
    while (getline(file, line)) {
        std::string cleaned;
        for (char c : line) {
            char upper = toupper(c);
            if (dnaToIndex(upper) != -1) cleaned += upper;
        }
        if (!cleaned.empty()) patterns.push_back(cleaned);
    }
}

void loadText(const char* filename, std::string& text) {
    std::ifstream file(filename);
    std::string line;
    while (getline(file, line)) {
        for (char c : line) {
            char upper = toupper(c);
            if (upper == 'A' || upper == 'C' || upper == 'G' || upper == 'T') {
                text += upper;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <dna_text_file> <pattern_file>\n", argv[0]);
        return 1;
    }

    std::vector<std::string> patterns;
    std::string text;
    loadPatterns(argv[2], patterns);
    loadText(argv[1], text);

    int (*h_gotoFn)[ALPHABET_SIZE] = new int[MAX_NODES][ALPHABET_SIZE];
    int* h_output = new int[MAX_NODES]();
    int statesCount = 0;
    buildFSM(patterns, h_gotoFn, h_output, statesCount);

    gpuErrchk(cudaMemcpyToSymbol(d_gotoFn, h_gotoFn, statesCount * ALPHABET_SIZE * sizeof(int)));
    gpuErrchk(cudaMemcpyToSymbol(d_output, h_output, statesCount * sizeof(int)));

    int textLength = text.length();
    char* d_text;
    int* d_matches;

    gpuErrchk(cudaMalloc(&d_text, textLength * sizeof(char)));
    gpuErrchk(cudaMalloc(&d_matches, textLength * sizeof(int)));

    gpuErrchk(cudaMemcpy(d_text, text.c_str(), textLength * sizeof(char), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_matches, 0, textLength * sizeof(int)));

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocks = (textLength + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(start));

    ahoCorasickParallelGPU<<<blocks, threadsPerBlock>>>(d_text, textLength, d_matches);
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    gpuErrchk(cudaEventElapsedTime(&elapsed_ms, start, stop));

    printf("============================================================\n");
    printf("Pattern file   : %s\n", argv[2]);
    printf("Pattern count  : %zu\n", patterns.size());
    printf("Text length    : %d\n", textLength);
    printf("FSM States     : %d\n", statesCount);
    printf("Search time    : %.2f ms (GPU parallel kernel)\n", elapsed_ms);
    printf("============================================================\n");

    // Optional: copy and sum matches
    std::vector<int> h_matches(textLength);
    gpuErrchk(cudaMemcpy(h_matches.data(), d_matches, textLength * sizeof(int), cudaMemcpyDeviceToHost));

    long long totalMatches = 0;
    for (int m : h_matches) totalMatches += m;
    printf("Total matches found: %lld\n", totalMatches);

    printf("Running CPU validation...\n");

        long long cpuMatches = 0;
        for (int i = 0; i < text.size(); ++i) {
            for (const std::string& pat : patterns) {
                if (i + pat.size() <= text.size() &&
                    text.compare(i, pat.size(), pat) == 0) {
                    cpuMatches++;
                }
            }
        }

        if (cpuMatches == totalMatches) {
            printf(" CPU-GPU match successful! Total matches: %lld\n", cpuMatches);
        } else {
            printf(" Mismatch detected!\n");
            printf("    CPU matches: %lld\n", cpuMatches);
            printf("    GPU matches: %lld\n", totalMatches);
        }
    // Cleanup
    cudaFree(d_text);
    cudaFree(d_matches);
    delete[] h_gotoFn;
    delete[] h_output;

    return 0;
}