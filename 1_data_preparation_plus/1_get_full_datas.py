import csv

# Step 1: Load chembl_uniprot_mapping.txt and build drug_target_dict
drug_target_dict = {}
with open('../0_database/0_database/chembl_uniprot_mapping.txt', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    next(reader)  # Skip header
    for row in reader:
        chembl_id = row[1]
        uniprot_id = row[0]
        drug_target_dict[chembl_id] = uniprot_id

# Step 2: Load target_sequences.fasta and build a dictionary of target sequences by Uniprot ID
target_seqs = {}
with open('../0_database/1_get_prot_seqs/target_sequences.fasta', 'r') as file:
    uniprot_id = None
    sequence = ''
    for line in file:
        if line.startswith('>'):
            if uniprot_id and sequence:
                target_seqs[uniprot_id] = sequence  # Save previous sequence
            uniprot_id = line.split('|')[1]  # Extract Uniprot ID
            sequence = ''
        else:
            sequence += line.strip()
    if uniprot_id and sequence:
        target_seqs[uniprot_id] = sequence  # Save last sequence

# Step 3: Open ChEMBL34_CTI_literature_only_full_dataset.csv and extract the required information
output_data = []
with open('../0_database/0_database/ChEMBL34_CTI_literature_only/ChEMBL34_CTI_literature_only_full_dataset.csv', 'r') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        if row['mutation']:  # Skip if mutation column is not empty
            continue

        canonical_smiles = row['canonical_smiles']
        parent_chembl_id = row['parent_chemblid']
        target_chembl_id = row['target_chembl_id']
        pchembl = row['pchembl_value_mean_BF']

        if not pchembl:
            continue

        # Look up Uniprot ID
        uniprot_id = drug_target_dict.get(target_chembl_id)
        if not uniprot_id:
            continue  # Skip if Uniprot ID is not found

        # Get target sequence from target_seqs dictionary
        target_seq = target_seqs.get(uniprot_id)
        if not target_seq:
            continue  # Skip if target sequence is not found

        # Append the required data
        output_data.append([parent_chembl_id, canonical_smiles, target_chembl_id, uniprot_id, target_seq, pchembl])

# Step 4: Save to ./1_get_full_datas/pre_database.csv
with open('./1_get_full_datas/pre_database.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['parent_chemblid', 'canonical_smiles', 'target_chembl_id', 'uniprot_id', 'target_seq', 'pchembl'])
    writer.writerows(output_data)
