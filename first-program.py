from Bio.Seq import Seq
from Bio import SeqIO

# Parse FASTA file
for record in SeqIO.parse("sequences.fasta", "fasta"):
    print(f"ID: {record.id}, Length: {len(record.seq)}")

# Reverse complement DNA
dna_seq = Seq("ATGCGATCGTA")
rev_comp = dna_seq.reverse_complement()
print(f"Original: {dna_seq}\nReverse Complement: {rev_comp}")
