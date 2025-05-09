The difference between these files are the file paths specifying the pattern and dna bank, as well as the DEFINES:  

```c
#define MAX_DNA 1000000  
#define MAX_PATTERNS 128  
#define MAX_PATTERN_LEN 10  
#define D 4  
```


  MAX_DNA to specify how much there is inside the bank  
  MAX_PATTERNS to specify how many patterns in the patterns txt  
  MAX_PATTERN_LEN to specify how long each pattern is  
  D to specify how many characters for the hash. Its 4 for all code because of A, C, G, T, which are mentioned in '__device__ int char_to_int'  

  After this, hange the location accordingly so that it located the pattern and dna bank txt files.  
  If not, the code will run but you will get "Segmentation Fault" error and nothing will happen  


    FILE *f = fopen("../../Dataset/dna_bank_1m.txt", "r");
    fgets(dna, MAX_DNA, f);
    fclose(f);
    int len = strlen(dna);

    f = fopen("../../Dataset/patterns_8_len10.txt", "r");
    int npat = 0;
    while (fgets(patterns[npat], MAX_PATTERN_LEN, f)) {
        patterns[npat][strcspn(patterns[npat], "\r\n")] = 0;
        npat++;
    }
    fclose(f);  

