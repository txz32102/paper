mkdir raw_data
wget https://ftp.ncbi.nlm.nih.gov/blast/db/FASTA/swissprot.gz --no-check-certificate -o raw_data/swissprot.gz
gunzip swissprot.gz