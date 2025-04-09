# 🔍 Parallel String Matching with CUDA

This project implements and compares two string matching algorithms, **Rabin-Karp** and **Aho-Corasick**, in both serial (CPU) and parallel (GPU) versions using **CUDA**. The goal is to evaluate their performance when applied to large-scale genomic data, specifically DNA sequences of *Mus musculus* (house mouse).

---

## 📌 Project Objectives
- To implement the **Rabin-Karp** and **Aho-Corasick** algorithms in both serial and parallel versions.
- To optimize and run the parallel versions using **CUDA** for GPU acceleration.
- To measure and compare the **execution time** and **memory usage** of both algorithms.
- To analyze which algorithm performs better under parallel processing on large DNA datasets.

---

## 🧬 Dataset
We use genomic sequence data (nucleotide) as the input text and pattern:
- Source: [NCBI Genome - Mus musculus](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)
- You can modify or generate custom pattern sequences to test scalability.

---

## 🛠️ Technologies Used
- **CUDA Toolkit** (GPU parallel programming)
- **C/C++** for algorithm implementation
- **NVIDIA Nsight Systems** for profiling
- **Makefile** or shell commands for execution
- **Git** and **GitHub** for version control and documentation

---

## 📁 Project Structure
```
.
├── aho_corasick_cuda/         # CUDA-based Aho-Corasick implementation
├── rabin_karp_cuda/           # CUDA-based Rabin-Karp implementation
├── serial_versions/           # CPU (non-parallel) versions of both algorithms
├── dataset/                   # Sample input DNA sequences and patterns
├── profiling_reports/         # Profiling results (.nsys-rep) and analysis
├── result_graphs/             # Graphs comparing execution time & memory
└── README.md                  # This file
```

---

## ▶️ How to Run

### Compile
```bash
# For Rabin-Karp
cd rabin_karp_cuda
make

# For Aho-Corasick
cd aho_corasick_cuda
make
```

### Execute
```bash
./rabin_karp input.txt pattern.txt
./aho_corasick input.txt pattern.txt
```

### Profiling
Run this command in the same directory as the compiled file:
```bash
nsys profile -o rk_profile ./rabin_karp input.txt pattern.txt
```

---

## 📊 Output & Analysis
After execution, profiling data such as:
- Total execution time
- Kernel performance
- Memory usage

...are collected using Nsight Systems and converted into tables and graphs. All results are analyzed in the final report.

---

## 📚 References
- Thambawita et al., *An Optimized Parallel Failure-less Aho-Corasick Algorithm for DNA Sequence Matching*, 2016.
- Shah et al., *Parallelizing Rabin-Karp Algorithm on GPU Using CUDA*, 2018.
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

## 👨‍💻 Author
Final Project - Parallel Computing  
Computer Science – Batch 2022  
Universitas Pelita Harapan

Callysa Tanjaya
Hans Adriel
Jovan Christian
Raden Jiwa Bumi Prajasantana

```

---

Silakan ganti `[Nama Kampusmu]` kalau kamu mau pakai nama aslimu. Kalau butuh versi dalam Bahasa Indonesia juga, tinggal bilang aja! 😄
