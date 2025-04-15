import os

fasta_file = "GCA_000001635.9_GRCm39_genomic.fna"
output_file = "dna_bank.txt"

print("✔ File exists:", os.path.exists(fasta_file))

if not os.path.exists(fasta_file):
    exit()

with open(fasta_file, 'r') as f_in, open(output_file, 'w') as f_out:
    line_count = 0
    written = 0
    for line in f_in:
        line_count += 1
        if line.startswith(">"):
            continue
        line = line.strip().upper()
        if set(line) == {'N'}:
            continue
        clean = ''.join(c for c in line if c in {'A', 'T', 'C', 'G'})
        f_out.write(clean)
        written += len(clean)
        if line_count % 50000 == 0:
            print(f"Lines processed: {line_count}, bases written: {written}")

print(f"✅ Done. Total bases written: {written}")
