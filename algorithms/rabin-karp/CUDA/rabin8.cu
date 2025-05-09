#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_DNA 1000000
#define MAX_PATTERNS 8
#define MAX_PATTERN_LEN 10
#define D 4

__device__ int char_to_int(char c) {
    if (c == 'A') return 0;
    if (c == 'C') return 1;
    if (c == 'G') return 2;
    if (c == 'T') return 3;
    return 0;
}

__global__ void findMatches(char *text, int len, char *pattern, int pat_len, int p0, int *result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > len - pat_len) return;

    int hash = 0;
    for (int j = 0; j < pat_len; ++j) {
        hash = hash * D + char_to_int(text[i + j]);
    }

    if (hash == p0) {
        bool match = true;
        for (int j = 0; j < pat_len; j++) {
            if (text[i + j] != pattern[j]) {
                match = false;
                break;
            }
        }
        if (match) result[i] = 1;
    }
}

int compute_hash(const char *p, int len) {
    int h = 0;
    for (int i = 0; i < len; i++) {
        h = h * D + (p[i] == 'A' ? 0 : p[i] == 'C' ? 1 : p[i] == 'G' ? 2 : 3);
    }
    return h;
}

int main() {
    static char dna[MAX_DNA];
    static char patterns[MAX_PATTERNS][MAX_PATTERN_LEN];
    static int result[MAX_DNA];

    FILE *f = fopen("dna_bank_1m.txt", "r");
    fgets(dna, MAX_DNA, f);
    fclose(f);
    int len = strlen(dna);

    f = fopen("patterns_8_len10.txt", "r");
    int npat = 0;
    while (fgets(patterns[npat], MAX_PATTERN_LEN, f)) {
        patterns[npat][strcspn(patterns[npat], "\r\n")] = 0;
        npat++;
    }
    fclose(f);

    char *d_dna, *d_pattern;
    int *d_result;

    cudaMalloc(&d_dna, len);
    cudaMemcpy(d_dna, dna, len, cudaMemcpyHostToDevice);
    cudaMalloc(&d_result, len * sizeof(int));

    for (int i = 0; i < npat; i++) {
        int pat_len = strlen(patterns[i]);
        int hash = compute_hash(patterns[i], pat_len);
        cudaMalloc(&d_pattern, pat_len);
        cudaMemcpy(d_pattern, patterns[i], pat_len, cudaMemcpyHostToDevice);
        cudaMemset(d_result, 0, len * sizeof(int));

        int blocks = (len + 255) / 256;
        findMatches<<<blocks, 256>>>(d_dna, len, d_pattern, pat_len, hash, d_result);
        cudaMemcpy(result, d_result, len * sizeof(int), cudaMemcpyDeviceToHost);

        // Commented out print statements for performance profiling
        // printf("Pattern %d: %s\n", i + 1, patterns[i]);
        // for (int j = 0; j < len; j++) {
        //     if (result[j]) printf("  Match at position %d\n", j);
        // }

        cudaFree(d_pattern);
    }

    cudaFree(d_dna);
    cudaFree(d_result);

    // Commented out file I/O for performance profiling
    // printf("Done!\n");
    // FILE *out = fopen("output.txt", "w");
    // fprintf(out, "Done!\n");
    // fclose(out);

    return 0;
}
