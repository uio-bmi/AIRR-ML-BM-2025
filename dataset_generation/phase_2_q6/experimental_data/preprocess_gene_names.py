import re
import os.path
import numpy as np
import pandas as pd

def perform_gene_name_replacement(df):
    """
    Since Q6 involves using positive and negative examples from different datasets generated using different protocols,
    we need to ensure that the gene names are consistent across all datasets. This function performs gene name
    replacement for the given DataFrame such that gene names are consistent across examples.

    """
    df_cleaned = df[~df['amino_acid'].str.contains(r'^na$|[*]')]
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'/.*', '', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'\*.*', '', regex=True)
    df_cleaned['j_resolved'] = df_cleaned['j_resolved'].str.replace(r'/.*', '', regex=True)
    df_cleaned['j_resolved'] = df_cleaned['j_resolved'].str.replace(r'\*.*', '', regex=True)

    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TCRBV', 'TRBV', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'(?<!\d)0+(\d+)', r'\1', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'-(?!0)(\d)', r'-\1', regex=True)

    df_cleaned['j_resolved'] = df_cleaned['j_resolved'].str.replace(r'TCRBJ', 'TRBJ', regex=True)
    df_cleaned['j_resolved'] = df_cleaned['j_resolved'].str.replace(r'(?<!\d)0+(\d+)', r'\1', regex=True)
    df_cleaned['j_resolved'] = df_cleaned['j_resolved'].str.replace(r'-(?!0)(\d)', r'-\1', regex=True)

    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TRBV6-\d+', 'TRBV6', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TRBV1-\d+', 'TRBV1', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TRBV12-\d+', 'TRBV12', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TRBV7-\d+', 'TRBV7', regex=True)
    df_cleaned['v_resolved'] = df_cleaned['v_resolved'].str.replace(r'TRBV11-\d+', 'TRBV11', regex=True)

    df_cleaned = gene_name_replacement(df_cleaned, all_datasets_replacements)
    df_cleaned = df_cleaned.drop_duplicates()
    return df_cleaned


emerson_replacements = {'TRBV13': 'TRBV13-1',
                        'TRBV14': 'TRBV14-1',
                        'TRBV15': 'TRBV15-1',
                        'TRBV16': 'TRBV16-1',
                        'TRBV17': 'TRBV17-1',
                        'TRBV18': 'TRBV18-1',
                        'TRBV19': 'TRBV19-1',
                        'TRBV2': 'TRBV2-1',
                        'TRBV23': 'TRBV23-1',
                        'TRBV24': 'TRBV24-1',
                        'TRBV25': 'TRBV25-1',
                        'TRBV26': 'TRBV26-1',
                        'TRBV27': 'TRBV27-1',
                        'TRBV28': 'TRBV28-1',
                        'TRBV29': 'TRBV29-1',
                        'TRBV3': 'TRBV3-1',
                        'TRBV30': 'TRBV30-1',
                        'TRBV4': 'TRBV4-1',
                        'TRBV9': 'TRBV9-1'}

all_datasets_replacements = {
    'TRBVA-or9_2': 'TRBVA',
    'TRBV29-or9_2': 'TRBV29-1',
    'TRBV26-or9_2': 'TRBV26-1',
    'TRBV25-or9_2': 'TRBV25-1',
    'TRBV23-or9_2': 'TRBV23-1',
    'TRBV22-or9_2': 'TRBV22-1',
    'TRBV21-or9_2': 'TRBV21-1',
    'TRBV20-or9_2': 'TRBV20',
}


def gene_name_replacement(df, v_replacements):
    df['v_resolved'] = df['v_resolved'].replace(v_replacements)
    return df

def _replace_trb_gene(v_gene_col, j_gene_col):
    return re.sub(r"^TRBJ.*|''", j_gene_col, v_gene_col)


def process_covid_data(metadata_tsv, out_path):
    metadata = pd.read_csv(metadata_tsv, sep='\t', header=0, index_col=False)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    source_path = os.path.dirname(metadata_tsv)
    metadata = metadata[(metadata['total_rearrangements'] >= 50000) & (metadata['total_rearrangements'] <= 140000)]
    metadata['filename'] = metadata['sample_name'] + '.tsv'
    metadata.reset_index(drop=True, inplace=True)
    for i in range(len(metadata)):
        print("Processing file", i)
        original_filename = metadata.loc[i, "filename"]
        source_file = os.path.join(source_path, original_filename)
        destination_file = os.path.join(out_path, metadata.loc[i, "filename"])
        relevant_cols = ["amino_acid", "v_resolved", "j_resolved", "templates"]
        try:
            df = pd.read_csv(source_file, header=0, sep='\t', usecols=relevant_cols)
            df = df[relevant_cols]
            df = perform_gene_name_replacement(df)
            df.columns = ["junction_aa", "v_call", "j_call", "templates"]
            df.to_csv(destination_file, sep='\t', index=False)
        except Exception as e:
            continue
    metadata.to_csv(os.path.join(out_path, "metadata.csv"), index=False)


def process_emerson_data(metadata_csv, out_path):
    metadata = pd.read_csv(metadata_csv)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    source_path = os.path.dirname(metadata_csv)
    for i in range(len(metadata)):
        print("Processing file", i)
        original_filename = metadata.loc[i, "filename"]
        source_file = os.path.join(source_path, "repertoires", original_filename)
        destination_file = os.path.join(out_path, metadata.loc[i, "filename"])
        relevant_cols = ["junction_aa", "v_genes", "j_genes", "v_subgroups", "duplicate_count"]
        try:
            rep_df = pd.read_csv(source_file, header=0, sep='\t', index_col=False, usecols=relevant_cols)
            rep_df[['v_subgroups', "v_genes"]] = rep_df[['v_subgroups', "v_genes"]].fillna('unknown')
            rep_df['v_genes'] = np.vectorize(_replace_trb_gene)(rep_df["v_genes"], rep_df["v_subgroups"])
            rep_df = rep_df[['junction_aa', 'v_genes', 'j_genes', 'duplicate_count']]
            rep_df.columns = ["amino_acid", "v_resolved", "j_resolved", "templates"]
            rep_df = perform_gene_name_replacement(rep_df)
            rep_df = gene_name_replacement(rep_df, emerson_replacements)
            rep_df.columns = ["junction_aa", "v_call", "j_call", "templates"]
            rep_df.to_csv(destination_file, sep='\t', index=False)
        except Exception as e:
            continue
    metadata.to_csv(os.path.join(out_path, "metadata.csv"), index=False)


def process_t1d_data(metadata, source_path, destination_path):
    metadata = pd.read_csv(metadata)
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    for i in range(len(metadata)):
        print("Processing file", i)
        original_filename = metadata.loc[i, "original_filename"]
        dataset = metadata.loc[i, "dataset"]
        source_file = os.path.join(source_path, f"cohort_{dataset}", original_filename)
        destination_file = os.path.join(destination_path, metadata.loc[i, "original_filename"])
        relevant_cols = ["amino_acid", "v_gene", "j_gene", "templates"]
        df = pd.read_csv(source_file, header=0, sep='\t', usecols=relevant_cols)
        df = df[relevant_cols]
        df.columns = ["amino_acid", "v_resolved", "j_resolved", "templates"]
        df = perform_gene_name_replacement(df)
        df.columns = ["junction_aa", "v_call", "j_call", "templates"]
        df.to_csv(destination_file, sep='\t', index=False)
    metadata.to_csv(os.path.join(destination_path, "metadata.csv"), index=False)


## usage example #
# if __name__ == '__main__':
#     process_covid_data(metadata_tsv='/path/to/covid_metadata.tsv',
#                        out_path='/path/to/covid_cleaned_data')
#     process_emerson_data(metadata_csv="/path/to/cmv_metadata.csv",
#                          out_path="/path/to/cmv_cleaned_data")
#     process_t1d_data(metadata="/path/to/q6_t1d_metadata.csv",
#                      source_path="/path/to/t1d/dataset/",
#                      destination_path="/path/to/t1d_cleaned_data/")
