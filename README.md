#  Parallel String Matching with CUDA

Link Google Docs : https://docs.google.com/document/d/1UJy5XhpXYB6y1YbmfB4oBHEgdm9OmMm5oTTPSiQlqOU/edit?usp=sharing

This project implements and compares two string matching algorithms, **Rabin-Karp** and **Aho-Corasick**, in both serial (CPU) and parallel (GPU) versions using **CUDA**. The goal is to evaluate their performance when applied to large-scale genomic data, specifically DNA sequences of *Mus musculus* (house mouse).

---

##  Project Objectives
- To implement the **Rabin-Karp** and **Aho-Corasick** algorithms in both serial and parallel versions.
- To optimize and run the parallel versions using **CUDA** for GPU acceleration.
- To measure and compare the **execution time** and **memory usage** of both algorithms.
- To analyze which algorithm performs better under parallel processing on large DNA datasets.

---

##  Dataset
We use genomic sequence data (nucleotide) as the input text and pattern:
- Source: [NCBI Genome - Mus musculus](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)
- Datasets and Patterns (from 8to 1024) already included inside the 'Dataset' folder

---

##  Technologies Used
- **CUDA Toolkit** (GPU parallel programming)
- **C/C++** for algorithm implementation
- **NVIDIA Nsight Systems** for profiling
- **Makefile** or shell commands for execution
- **Git** and **GitHub** for version control and documentation
- **Cisco AnnyConnect VPN** and **SSH/HPC Server** specifically for execution of our code FOR the essay project. Our code can be executed locally by anyone (needs NVIDIA GPU for .cu files)

---

##  Project Structure
```
.
├── Dataset/                   # Contains DNA text files and pattern sets
├── Documents/                 # Contains reports, images, and Nsight profiling data
│   ├── Nsys Aho/              # Nsight Systems report for Aho-Corasick (CUDA)
│   └── Nsys Rabin/            # Nsight Systems report for Rabin-Karp (CUDA)
├── algorithms/                # Contains folders for implementation of both algorithms
│   ├── aho-corasick/          # Aho-Corasick algorithm (C++ and CUDA versions both here)
│   └── rabin-karp/            # Rabin-Karp algorithm. The cpp version is here
│       └── CUDA/              # CUDA version of Rabin-Karp algorithm is here
└── README.md                  # **YOU ARE CURRENTLY HERE**

```

---

##  How to Run

### Compile
```bash
# For Rabin-Karp non-CUDA
cd rabin_karp              # Change location to where the code is
g++ -o rb_sequential rb_sequential.cpp

# For Rabin-Karp CUDA (repeat from 8 until 1024, or whichever you want)
cd CUDA              # Change location to where the code is
nvcc -o rabin1024 rabin1024.cu  

# For Aho-Corasick non-CUDA
cd aho-corasick              # Change location to where the code is
g++ -o ac_sequential ac_sequential.cpp

# For Aho-Corasick CUDA
cd aho-corasick             # Change location to where the code is  
nvcc -o aho ac.cu

```

### Execute
```bash
# Rabin-Karp non-CUDA (just ./[NAME OF COMPILED EXE])
./rb_sequential

# Rabin-Karp CUDA (just change 'rabin1024' to the name of the compiled .exe you just did above)
./rabin1024
# If that doesnt work, this:
rabin1024.exe

# Aho-Corasick non-CUDA (just ./[NAME OF COMPILED EXE])
./ac_sequential

# Aho-Corasick CUDA (./[NAME OF COMPILED FILE] [NAME OF DNA BANK TXT] [NAME OF PATTERNS TXT]
./aho dna_bank_1m.txt patterns_8_len10.txt
./aho dna_bank_1m.txt patterns_16_len10.txt
./aho dna_bank_1m.txt patterns_32_len10.txt
./aho dna_bank_1m.txt patterns_64_len10.txt
./aho dna_bank_1m.txt patterns_128_len10.txt
./aho dna_bank_1m.txt patterns_256_len10.txt
./aho dna_bank_1m.txt patterns_512_len10.txt
./aho dna_bank_1m.txt patterns_1024_len10.txt
```

### Profiling
Run this command in the same directory as the compiled file:
```bash
nsys profile -o [INSERT OUTPUT NAME] --stats=true [EXECUTE COMMAND SUCH AS: ./aho dna_bank_1m.txt patterns_8_len10.txt]
# EXAMPLES:
nsys profile -o report_rabin1024 --stats=true ./rabin1024
nsys profile -o aho_L8 --stats=true ./aho dna_bank_1m.txt patterns_8_len10.txt  
```

---

##  Output & Analysis (WIP, below are placeholders)
After execution, profiling data such as:
- Total execution time
- Kernel performance
- Memory usage

...are collected using Nsight Systems and converted into tables and graphs. All results are analyzed in the final report.

---

##  References (not final yet)
- Thambawita et al., *An Optimized Parallel Failure-less Aho-Corasick Algorithm for DNA Sequence Matching*, 2016.
- Shah et al., *Parallelizing Rabin-Karp Algorithm on GPU Using CUDA*, 2018.
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

##  Author
Final Project - Parallel Computing  
Computer Science – Batch of 2022  
Universitas Pelita Harapan

Callysa Tanjaya  
Hans Adriel  
Jovan Christian  
Raden Jiwa Bumi Prajasantana  

## License  
```
MIT License. Check here: https://opensource.org/licenses/MIT
TL;DR = Allowed to modify, distribute, and even commercial usage as long as there is ATTRIBUTION
