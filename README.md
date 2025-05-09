#  Parallel String Matching with CUDA  

## Our Project

Link to Google Docs : https://docs.google.com/document/d/1UJy5XhpXYB6y1YbmfB4oBHEgdm9OmMm5oTTPSiQlqOU/edit?usp=sharing  
Link to Spreadsheet : https://docs.google.com/spreadsheets/d/1MurFNFRxrK0nIjqRHkG9ItVnqqXn7qz4/edit?usp=sharing&ouid=100613193120402018407&rtpof=true&sd=true  

This project implements and compares two string matching algorithms, **Rabin-Karp** and **Aho-Corasick**, in both serial only (CPU, denoted as .cpp) and parallel (GPU, using CUDA and denoted as .cu) versions.  
The goal is to evaluate their performance when applied to large-scale genomic data, specifically DNA sequences of *Mus musculus* (house mouse).  


##  Project Objectives
- To implement the Rabin-Karp and Aho-Corasick algorithms in both serial and parallel versions.
- To utilize GPU acceleration by running both alrogithms using CUDA.
- To measure and compare the **execution time** and **memory usage** of both algorithms.
- To analyze the results; compare which algorithm performs better under parallel processing on large DNA datasets.

---

##  Dataset
We use genomic sequence data (nucleotide) as the input text and pattern:
- Source: [NCBI Genome - Mus musculus](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)
- Cleaned dataset and specified patterns (from 8 to 1024) are already included inside the 'Dataset' folder
- You can use your own datasets and patterns. Just change the algorithm's code's inputs to recognize it.

---

##  Technologies Used
- **CUDA Toolkit** for GPU parallel programming; compile and execute CUDA code  
- **C/C++** for algorithm implementation  
- **NVIDIA Nsight Systems** for profiling  
- **GCC** to compile C++ files  
- **Command Prompt, Terminal or PowerShell** for compiling and execution  
- **Git** and **GitHub** for version control and documentation  
- **Cisco AnnyConnect VPN** and **SSH/HPC Server** specifically for execution of our code FOR the essay project. Our code can be executed locally by anyone (needs NVIDIA GPU for .cu files)
- **WinSCP** to view and manage files on the SSH/HPC server more beautifully and more convenient. Not required if not using SSH/HPC server
- **Visual Studio Code**  as our members' choice for text editors for coding, as well as pushing updates to Github
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
└── README.md                  # YOU ARE CURRENTLY HERE!

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

##  Output & Analysis  
After execution, profiling data such as:
- Total execution time
- Kernel performance
- Memory usage

...are collected using Nsight Systems (nsys) and recorded onto tables and graphs. All results are analyzed in the final report doc and spreadsheet.  
'nsys example.png' image inside Documents folder shows what the reports looked like when we analyzed with Nsight Systems (nsys) user interface.

---

##  References    
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [GCC, the GNU Compiler Collection](https://gcc.gnu.org/)
- [Git](https://git-scm.com/)
- [Cisco AnyConnect Secure Mobility Client](https://www.cisco.com/c/en/us/support/security/anyconnect-secure-mobility-client-v4-x/model.html) (This was only used to access our university's SSH/HPC server. Obtain this software from your institution or workplace, or just the code run on your own machine)
- [WinSCP](https://winscp.net/eng/download.php)

### Academic References:  
```
[1]	V. Thambawita, R. Ragel, and D. Elkaduwe, “An optimized Parallel Failure-less Aho-Corasick algorithm for DNA sequence matching,” Dec. 2016, pp. 1–6. doi: 10.1109/ICIAFS.2016.7946533.
[2]	P. Shah and R. Oza, “Improved Parallel Rabin-Karp Algorithm Using Compute Unified Device Architecture,” vol. 84, 2018, pp. 236–244. doi: 10.1007/978-3-319-63645-0_26.
[3]	J. Ghorpade, “GPGPU Processing in CUDA Architecture,” Adv. Comput. Int. J., vol. 3, no. 1, pp. 105–120, Jan. 2012, doi: 10.5121/acij.2012.3109.
[4]	I. Mocanu, “An INTRODUCTION TO CUDA Programming,” J. Inf. Syst. Oper. Manag., vol. 2, pp. 495–506, Jan. 2008.
[5]	S. Kanda, K. Akabe, and Y. Oda, “Engineering faster double-array Aho-Corasick automata,” Softw. Pract. Exp., vol. 53, no. 6, pp. 1332–1361, Jun. 2023, doi: 10.1002/spe.3190.
[6]	A. Abbas, M. Fayez, and H. Khaled, “Multi-Pattern GPU Accelerated Collision-Less Rabin-Karp for NIDS,” Int. J. Distrib. Syst. Technol. IJDST, vol. 15, no. 1, pp. 1–16, 2024, doi: 10.4018/IJDST.341269.
[7]	A. V. Aho and M. J. Corasick, “Efficient string matching: an aid to bibliographic search,” Commun. ACM, vol. 18, no. 6, pp. 333–340, Jun. 1975, doi: 10.1145/360825.360855.
[8]	P. Mahmud, A. Rahman, and K. Hasan Talukder, “An Improved Hashing Approach for Biological Sequence to Solve Exact Pattern Matching Problems,” Appl. Comput. Intell. Soft Comput., vol. 2023, no. 1, p. 3278505, 2023, doi: 10.1155/2023/3278505.
[9]	A. P. Gope and R. N. Behera, “A Novel Pattern Matching Algorithm in Genome Sequence Analysis,” vol. 5, 2014.
[10]	“Improved Parallel Rabin-Karp Algorithm Using Compute Unified Device Architecture,” ar5iv. Accessed: May 09, 2025. [Online]. Available: https://ar5iv.labs.arxiv.org/html/1810.01051
```

---
##  Authors
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
