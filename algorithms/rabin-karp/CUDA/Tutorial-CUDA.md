The difference between these files are the file paths specifying the pattern and dna bank:  
  
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

Chane the location accordingly so that it located the pattern and dna bank txt files. If not, the code will run but you will get "Segmentation Fault" error
