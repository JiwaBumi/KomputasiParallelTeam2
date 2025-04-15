import random

def extract_dna_sequence(fasta_file, output_file="dna_bank.txt"):
    print(f"Extracting DNA sequence from {fasta_file}...")
    with open(fasta_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.startswith('>'):
                f_out.write(line.strip().upper())
    print(f"DNA bank saved to {output_file}.")

def generate_patterns(dna_file="dna_bank.txt", pattern_lengths=[10], pattern_counts=[16, 32, 64, 128]):
    with open(dna_file, 'r') as f:
        dna = f.read().strip()
    
    for count in pattern_counts:
        for length in pattern_lengths:
            output_file = f"patterns_{count}_len{length}.txt"
            print(f"Generating {count} patterns of length {length} → {output_file}")
            with open(output_file, 'w') as f_out:
                for _ in range(count):
                    start = random.randint(0, len(dna) - length)
                    f_out.write(dna[start:start + length] + "\n")
    print("All pattern sets generated.")

# === Run the preparation ===
fasta_input = "GCA_000001635.9_GRCm39_genomic.fna"  # Update if filename is different
extract_dna_sequence(fasta_input)
generate_patterns()
