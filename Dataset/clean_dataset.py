def extract_clean_dna(fasta_file="GCA_000001635.9_GRCm39_genomic.fna", output_file="dna_bank_clean.txt", length=100000):
    allowed_bases = {'A', 'T', 'C', 'G'}
    sequence = ""

    print(f"🔍 Reading from: {fasta_file}")
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                continue  # Skip FASTA headers
            line = line.strip().upper()
            if not set(line).issubset({'A', 'T', 'C', 'G', 'N'}):
                continue  # Skip invalid lines
            if set(line) == {'N'}:
                continue  # Skip lines with only N
            # Remove N and unwanted characters
            cleaned = ''.join([c for c in line if c in allowed_bases])
            sequence += cleaned
            if len(sequence) >= length:
                break

    if len(sequence) < length:
        print(f"⚠️ Only extracted {len(sequence)} clean bases (target was {length})")
    else:
        print(f"✅ Successfully extracted {length} clean DNA bases")

    with open(output_file, 'w') as out:
        out.write(sequence[:length])
        print(f"📁 Saved to: {output_file}")


# === Run script ===
if __name__ == "__main__":
    extract_clean_dna()
