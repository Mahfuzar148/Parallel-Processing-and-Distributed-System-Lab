
/**
 * mpicxx MPI_Phonebook_search.cpp  -o search_mpi
 * mpirun -np 4 ./search_mpi "017 54"
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <mpi.h>

using namespace std;

#define MAX_LINE_LEN 300

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            cerr << "Usage: mpirun -np <procs> " << argv[0] << " <search_term>\n";
        }
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[1];

    vector<string> all_lines;
    int total_lines = 0;

    // Root loads file
    if (rank == 0) {
        ifstream file("/home/mahfuzar23/Parallel processing/phonebook1.txt");
        if (!file) {
            cerr << "[Rank 0] File not found\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string ln;
        while (getline(file, ln)) {
            if (!ln.empty()) all_lines.push_back(ln);
        }
        file.close();
        total_lines = all_lines.size();
        cout << "[Rank 0] Loaded " << total_lines << " lines.\n";
    }

    MPI_Bcast(&total_lines, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (total_lines == 0) {
        MPI_Finalize();
        return 0;
    }

    // Divide lines
    int lines_per_proc = total_lines / size;
    int extra = total_lines % size;
    int local_start = rank * lines_per_proc + min(rank, extra);
    int local_count = lines_per_proc + (rank < extra ? 1 : 0);

    char* local_lines = nullptr;
    if (local_count > 0) {
        local_lines = new char[local_count * MAX_LINE_LEN];
    }

    if (rank == 0) {
        char* sendbuf = new char[total_lines * MAX_LINE_LEN];
        for (int i = 0; i < total_lines; ++i) {
            strncpy(sendbuf + i * MAX_LINE_LEN, all_lines[i].c_str(), MAX_LINE_LEN - 1);
            sendbuf[i * MAX_LINE_LEN + MAX_LINE_LEN - 1] = '\0';
        }

        vector<int> sendcounts(size), displs(size);
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            int cnt = lines_per_proc + (r < extra ? 1 : 0);
            sendcounts[r] = cnt * MAX_LINE_LEN;
            displs[r] = offset;
            offset += sendcounts[r];
        }

        MPI_Scatterv(sendbuf, sendcounts.data(), displs.data(), MPI_CHAR,
                     local_lines, local_count * MAX_LINE_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

        delete[] sendbuf;
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_CHAR,
                     local_lines, local_count * MAX_LINE_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
    }

    // Local search
    vector<pair<int, string>> local_results;
    for (int i = 0; i < local_count; ++i) {
        char* line = local_lines + i * MAX_LINE_LEN;
        string s(line);
        if (s.find(search_term) != string::npos) {
            int global_index = local_start + i + 1;
            local_results.push_back({global_index, s});
        }
    }

    delete[] local_lines;

    // Gather counts
    int local_num = local_results.size();
    vector<int> all_num(size);
    MPI_Gather(&local_num, 1, MPI_INT, all_num.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root collects and saves to file + prints
    if (rank == 0) {
        vector<pair<int, string>> final_results;
        int total = 0;
        for (int c : all_num) total += c;
        final_results.reserve(total);

        // Rank 0's results
        final_results.insert(final_results.end(), local_results.begin(), local_results.end());

        // Other ranks
        for (int r = 1; r < size; ++r) {
            if (all_num[r] == 0) continue;

            vector<int> idx_buf(all_num[r]);
            MPI_Recv(idx_buf.data(), all_num[r], MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (int k = 0; k < all_num[r]; ++k) {
                int len;
                MPI_Recv(&len, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                char* buf = new char[len];
                MPI_Recv(buf, len, MPI_CHAR, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                string s(buf);
                delete[] buf;
                final_results.push_back({idx_buf[k], s});
            }
        }

        // Sort by index
        sort(final_results.begin(), final_results.end());

        // Print on screen
        cout << "\nResults for \"" << search_term << "\":\n";
        if (final_results.empty()) {
            cout << "No matches found.\n";
        } else {
            cout << "Found " << final_results.size() << " lines (sorted by index):\n\n";
            for (const auto& p : final_results) {
                cout << p.second << endl;
            }
        }

        // Save to output.txt
        ofstream outfile("output.txt");
        if (!outfile) {
            cerr << "Cannot create output.txt\n";
        } else {
            outfile << "Search term: " << search_term << "\n";
            outfile << "Total matches: " << final_results.size() << "\n\n";
            for (const auto& p : final_results) {
                outfile << p.second << "\n";
            }
            outfile.close();
            cout << "\nResults also saved to output.txt\n";
        }
    } else {
        // Send to root
        if (local_num > 0) {
            vector<int> idx_vec(local_num);
            for (int i = 0; i < local_num; ++i) idx_vec[i] = local_results[i].first;

            MPI_Send(idx_vec.data(), local_num, MPI_INT, 0, 0, MPI_COMM_WORLD);

            for (int i = 0; i < local_num; ++i) {
                const string& s = local_results[i].second;
                int len = s.length() + 1;
                MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                MPI_Send(s.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
