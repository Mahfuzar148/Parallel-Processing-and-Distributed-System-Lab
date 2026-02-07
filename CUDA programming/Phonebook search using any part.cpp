/**
!nvcc -arch=compute_75 -code=sm_75 search_any_part_sorted.cu -o search_any_sorted
!time ./search_any_sorted TANBIR 256
*/




%%writefile search_any_part_sorted.cu
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

#define MAX_LINE_LEN 300

__device__ bool has_substring(const char* text, const char* pat, int pat_len) {
    if (pat_len == 0) return true;
    for (int i = 0; text[i] != '\0'; ++i) {
        int j = 0;
        while (text[i + j] != '\0' && j < pat_len && text[i + j] == pat[j]) ++j;
        if (j == pat_len) return true;
    }
    return false;
}

__global__ void searchKernel(
    char* d_lines,
    int num_lines,
    char* d_pat,
    int pat_len,
    int* d_match
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lines) return;
    char* line = d_lines + idx * MAX_LINE_LEN;
    d_match[idx] = has_substring(line, d_pat, pat_len) ? 1 : 0;
}

// Comparator: index অনুযায়ী sort করার জন্য
struct LineWithIndex {
    string line;
    int index;
};

bool cmp(const LineWithIndex& a, const LineWithIndex& b) {
    return a.index < b.index;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <search_term> <threads_per_block>\n";
        return 1;
    }

    string search_term = argv[1];
    int threads = atoi(argv[2]);

    string fname = "/content/sample_data/phonebook1.txt";
    vector<string> original_lines;

    ifstream file(fname);
    if (!file) {
        cerr << "File not found: " << fname << endl;
        return 1;
    }

    string ln;
    while (getline(file, ln)) {
        if (!ln.empty()) original_lines.push_back(ln);
    }
    file.close();

    int n = original_lines.size();
    if (n == 0) {
        cerr << "No lines in file\n";
        return 1;
    }

    cout << "Loaded " << n << " lines.\n";

    char* h_lines = (char*)malloc(n * MAX_LINE_LEN);
    for (int i = 0; i < n; ++i) {
        strncpy(h_lines + i * MAX_LINE_LEN, original_lines[i].c_str(), MAX_LINE_LEN - 1);
        h_lines[i * MAX_LINE_LEN + MAX_LINE_LEN - 1] = '\0';
    }

    char *d_lines, *d_pat;
    int* d_match;

    int pat_len = search_term.length();

    cudaMalloc(&d_lines, n * MAX_LINE_LEN);
    cudaMalloc(&d_match, n * sizeof(int));
    cudaMalloc(&d_pat, pat_len + 1);

    cudaMemcpy(d_lines, h_lines, n * MAX_LINE_LEN, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pat, search_term.c_str(), pat_len + 1, cudaMemcpyHostToDevice);

    int blocks = (n + threads - 1) / threads;
    searchKernel<<<blocks, threads>>>(d_lines, n, d_pat, pat_len, d_match);

    cudaDeviceSynchronize();

    vector<int> match(n);
    cudaMemcpy(match.data(), d_match, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Collect results with their original index
    vector<LineWithIndex> results;
    for (int i = 0; i < n; ++i) {
        if (match[i]) {
            results.push_back({original_lines[i], i + 1});  // 1-based index
        }
    }

    // Sort by index (numerical order)
    sort(results.begin(), results.end(), cmp);

    // Print
    cout << "\nResults for \"" << search_term << "\":\n";
    if (results.empty()) {
        cout << "No matches found.\n";
    } else {
        cout << "Found " << results.size() << " lines (sorted by index):\n\n";
        for (const auto& r : results) {
            cout << r.line << endl;
        }
    }

    free(h_lines);
    cudaFree(d_lines);
    cudaFree(d_pat);
    cudaFree(d_match);

    return 0;
}
