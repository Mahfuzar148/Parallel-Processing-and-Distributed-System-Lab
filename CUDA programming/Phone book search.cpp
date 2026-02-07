
%%writefile search_phonebook_single_list.cu
#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;


#define MAX_STR_LEN 50


// Struct to hold one contact (name + number)
struct Contact {
    string name;
    string number;
};


// For sorting results
struct ResultContact {
    string name;
    string number;


    bool operator<(const ResultContact& other) const {
        return name < other.name;
    }
};


// Device substring check
__device__ bool check(const char* str1, const char* str2, int len) {
    for (int i = 0; str1[i] != '\0'; ++i) {
        int j = 0;
        while (str1[i + j] != '\0' && j < len && str1[i + j] == str2[j]) {
            ++j;
        }
        if (j == len) {
            return true;
        }
    }
    return false;
}


// CUDA kernel - same as before
__global__ void searchPhonebook(
    char* d_names,
    int num_contacts,
    char* search_name,
    int search_len,
    int* d_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;


    if (idx < num_contacts) {
        char* current_name = d_names + idx * MAX_STR_LEN;
        d_results[idx] = check(current_name, search_name, search_len) ? 1 : 0;
    }
}


int main(int argc, char* argv[]) {


    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <search_string> <threads_per_block>\n";
        return 1;
    }


    string search_string = argv[1];
    int threads_per_block = atoi(argv[2]);


    string file_name = "/content/sample_data/phonebook1.txt";


    // Single vector for all contacts
    vector<Contact> contacts;


    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return 1;
    }


    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;


        // Find the comma that separates name and number
        size_t pos = line.find("\",\"");
        if (pos == string::npos) continue;


        // Extract name (remove first and last quote)
        string name = line.substr(1, pos - 1);          // from after first " to before comma
        string number = line.substr(pos + 3, line.size() - pos - 4); // after "," to before last "


        contacts.push_back({name, number});
    }
    file.close();


    int num_contacts = contacts.size();
    if (num_contacts == 0) {
        cerr << "No contacts found.\n";
        return 1;
    }


    // Host memory for names only (for GPU search)
    char* h_names = (char*)malloc(num_contacts * MAX_STR_LEN);
    int* h_results = (int*)malloc(num_contacts * sizeof(int));


    // Copy only names to flat array for GPU
    for (int i = 0; i < num_contacts; ++i) {
        strncpy(h_names + i * MAX_STR_LEN,
                contacts[i].name.c_str(),
                MAX_STR_LEN - 1);
        h_names[i * MAX_STR_LEN + MAX_STR_LEN - 1] = '\0';
    }


    // Device memory
    char *d_names, *d_search_name;
    int* d_results;


    int search_len = search_string.length();


    cudaMalloc(&d_names, num_contacts * MAX_STR_LEN);
    cudaMalloc(&d_results, num_contacts * sizeof(int));
    cudaMalloc(&d_search_name, search_len + 1);


    cudaMemcpy(d_names, h_names,
               num_contacts * MAX_STR_LEN,
               cudaMemcpyHostToDevice);


    cudaMemcpy(d_search_name, search_string.c_str(),
               search_len + 1,
               cudaMemcpyHostToDevice);


    // Launch kernel
    int blocks = (num_contacts + threads_per_block - 1) / threads_per_block;
    searchPhonebook<<<blocks, threads_per_block>>>(
        d_names, num_contacts, d_search_name, search_len, d_results
    );


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
        return 1;
    }


    cudaDeviceSynchronize();


    // Get results back
    cudaMemcpy(h_results, d_results,
               num_contacts * sizeof(int),
               cudaMemcpyDeviceToHost);


    // Collect matching contacts
    vector<ResultContact> matched_contacts;
    for (int i = 0; i < num_contacts; ++i) {
        if (h_results[i] == 1) {
            matched_contacts.push_back({
                contacts[i].name,
                contacts[i].number
            });
        }
    }


    // Sort by name
    sort(matched_contacts.begin(), matched_contacts.end());


    // Print results
    cout << "\nSearch Results (Ascending Order):\n";
    if (matched_contacts.empty()) {
        cout << "No matches found.\n";
    } else {
        for (const auto& c : matched_contacts) {
            cout << c.name << " " << c.number << endl;
        }
    }


    // Cleanup
    free(h_names);
    free(h_results);
    cudaFree(d_names);
    cudaFree(d_results);
    cudaFree(d_search_name);


    return 0;
}
/**
!nvcc -arch=compute_75 -code=sm_75 search_phonebook_single_list.cu -o search_phonebook
!time ./search_phonebook TANBIR 256

*/
