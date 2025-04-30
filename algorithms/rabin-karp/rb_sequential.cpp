#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cmath>
using namespace std;
using namespace std::chrono;

const int d = 256;   // jumlah karakter yang mungkin (ASCII)
const int q = 101;   // bilangan prima untuk modulus hashing

vector<string> load_patterns(const string& filename) {
    vector<string> patterns;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        if (!line.empty())
            patterns.push_back(line);
    }
    return patterns;
}

string load_dna(const string& filename) {
    ifstream file(filename);
    string line, dna;
    while (getline(file, line)) {
        dna += line;
    }
    return dna;
}

int compute_hash(const string& str, int m) {
    int h = 0;
    for (int i = 0; i < m; i++) {
        h = (d * h + str[i]) % q;
    }
    return h;
}

void rabin_karp_batch_search(const string& text, const vector<string>& patterns) {
    if (patterns.empty()) return;

    int m = patterns[0].size();  // semua pattern panjang sama
    int n = text.size();

    unordered_multimap<int, const string*> hash_to_patterns;

    // Hash semua patterns sekali di awal
    for (const string& pat : patterns) {
        int h = compute_hash(pat, m);
        hash_to_patterns.insert({h, &pat});
    }

    int h = 1;
    for (int i = 0; i < m - 1; i++)
        h = (h * d) % q;

    int window_hash = 0;
    for (int i = 0; i < m; i++) {
        window_hash = (d * window_hash + text[i]) % q;
    }

    // Sliding window
    for (int i = 0; i <= n - m; i++) {
        // Jika hash match dengan salah satu pattern
        auto range = hash_to_patterns.equal_range(window_hash);
        for (auto it = range.first; it != range.second; ++it) {
            const string* pattern = it->second;
            if (equal(text.begin() + i, text.begin() + i + m, pattern->begin())) {
                // Pattern ditemukan di posisi i 
                // cout << "Found: " << *pattern << " at pos " << i << endl;
            }
        }

        // Update hash window berikutnya
        if (i < n - m) {
            window_hash = (d * (window_hash - text[i] * h) + text[i + m]) % q;
            if (window_hash < 0)
                window_hash += q;
        }
    }
}

int main() {
    string dna_file = "dna_bank_1m.txt";
    string dna = load_dna(dna_file);

    cout << "================= Rabin-Karp Batch Search =================" << endl;
    cout << "DNA file: " << dna_file << endl;
    cout << "DNA size: " << dna.size() << " characters" << endl << endl;

    for (int n = 3; n <= 10; n++) {
        int pattern_count = pow(2, n);
        string pattern_file = "patterns_" + to_string(pattern_count) + "_len10.txt";

        vector<string> patterns = load_patterns(pattern_file);
        if (patterns.empty()) {
            cout << "Pattern file " << pattern_file << " is empty or not found." << endl;
            continue;
        }

        auto start = high_resolution_clock::now();
        rabin_karp_batch_search(dna, patterns);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);

        cout << "------------------------------------------------------------" << endl;
        cout << "Pattern file   : " << pattern_file << endl;
        cout << "Pattern count  : " << pattern_count << endl;
        cout << "Pattern length : " << patterns[0].size() << endl;
        cout << "Search time    : " << duration.count() << " ms" << endl;
    }

    cout << "===================== Search Complete =====================" << endl;

    return 0;
}
