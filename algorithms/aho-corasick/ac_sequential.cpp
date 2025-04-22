#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

struct Node {
    unordered_map<char, Node*> children;
    Node* fail = nullptr;
    vector<string> outputs;
};

class AhoCorasick {
public:
    AhoCorasick(const vector<string>& patterns) {
        root = new Node();
        for (const string& pattern : patterns)
            insert(pattern);
        build();
    }

    void search(const string& text) {
        Node* node = root;
        for (int i = 0; i < text.length(); i++) {
            char ch = text[i];
            while (node != root && node->children.find(ch) == node->children.end())
                node = node->fail;
            if (node->children.count(ch))
                node = node->children[ch];
        }
    }

private:
    Node* root;

    void insert(const string& word) {
        Node* node = root;
        for (char ch : word) {
            if (!node->children.count(ch))
                node->children[ch] = new Node();
            node = node->children[ch];
        }
        node->outputs.push_back(word);
    }

    void build() {
        queue<Node*> q;
        for (auto& p : root->children) {
            p.second->fail = root;
            q.push(p.second);
        }

        while (!q.empty()) {
            Node* current = q.front(); q.pop();
            for (auto& p : current->children) {
                char ch = p.first;
                Node* child = p.second;

                Node* fallback = current->fail;
                while (fallback != root && fallback->children.find(ch) == fallback->children.end())
                    fallback = fallback->fail;
                if (fallback->children.count(ch))
                    child->fail = fallback->children[ch];
                else
                    child->fail = root;

                child->outputs.insert(child->outputs.end(),
                                      child->fail->outputs.begin(),
                                      child->fail->outputs.end());

                q.push(child);
            }
        }
    }
};

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

int main() {
    string dna_file = "dna_bank_1m.txt";
    string dna = load_dna(dna_file);

    for (int n = 3; n <= 10; n++) {
        int pattern_count = pow(2, n);
        string pattern_file = "patterns_" + to_string(pattern_count) + "_len10.txt";

        vector<string> patterns = load_patterns(pattern_file);

        AhoCorasick ac(patterns);

        auto start = high_resolution_clock::now();
        ac.search(dna);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        cout << "n=" << n 
             << " | patterns=" << pattern_count 
             << " | time=" << duration.count() << " ms" << endl;
    }

    return 0;
}
