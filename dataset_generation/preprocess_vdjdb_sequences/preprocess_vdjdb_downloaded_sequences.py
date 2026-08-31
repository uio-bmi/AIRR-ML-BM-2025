import pandas as pd

v_replacements = {'TRBV20': 'TRBV20-1',
                  'TRBV19-1': 'TRBV19',
                  'TRBV27-1': 'TRBV27',
                  'TRBV28-1': 'TRBV28',
                  'TRBV2-1': 'TRBV2',
                  'TRBV9-1': 'TRBV9',
                  'TRBV30-1': 'TRBV30',
                  'TRBV18-1': 'TRBV18',
                  'TRBV15-1': 'TRBV15',
                  'TRBV14-1': 'TRBV14',
                  'TRBV13-1': 'TRBV13',
                  'TRBV16-1': 'TRBV16',
                  'TRBV5-3': 'TRBV5',
                  'TRBV12-2': 'TRBV12',
                  'TRBV12-1': 'TRBV12',
                  'TRBV6-7': 'TRBV6',
                  'TRBV7-5': 'TRBV7',
                  'TRBV5-7': 'TRBV5'}


def gene_name_replacement(df, v_gene_replacements):
    df['V'] = df['V'].replace(v_gene_replacements)
    return df


def clean_vdjdb_seqs(input_file, split_df=False):
    df = pd.read_csv(input_file, sep='\t', na_values='')
    df = df[(df['Gene'] == 'TRB') & (df['Species'] == 'HomoSapiens') & (df['MHC class'] == 'MHCI')]
    df = df.drop_duplicates(subset=['CDR3'])
    df['V'] = df['V'].str.replace(r'/.*', '', regex=True)
    df['V'] = df['V'].str.replace(r'\*.*', '', regex=True)
    df['J'] = df['J'].str.replace(r'/.*', '', regex=True)
    df['J'] = df['J'].str.replace(r'\*.*', '', regex=True)
    required_columns = ['CDR3', 'V', 'J']
    df = df[required_columns]
    df = gene_name_replacement(df, v_replacements)
    df = df.sample(frac=1).reset_index(drop=True)
    if split_df:
        split_df_parts(df, input_file)
    else:
        output_file = input_file.replace('.tsv', '_processed.tsv')
        df.to_csv(output_file, sep='\t', index=False, header=False)


def split_df_parts(df, input_file):
    n = len(df)
    n_third = n // 3
    df1 = df.iloc[:n_third]
    df2 = df.iloc[n_third:2 * n_third]
    df3 = df.iloc[2 * n_third:]
    output_file1 = input_file.replace('.tsv', '_part1.tsv')
    output_file2 = input_file.replace('.tsv', '_part2.tsv')
    output_file3 = input_file.replace('.tsv', '_part3.tsv')
    df1.to_csv(output_file1, sep='\t', index=False, header=False)
    df2.to_csv(output_file2, sep='\t', index=False, header=False)
    df3.to_csv(output_file3, sep='\t', index=False, header=False)


def split_file_n_parts(input_file, n):
    df = pd.read_csv(input_file, sep='\t', header=None)
    n_rows = len(df)
    n_rows_per_part = n_rows // n
    for i in range(n):
        start = i * n_rows_per_part
        end = (i + 1) * n_rows_per_part
        if i == n - 1:
            end = n_rows
        df_part = df.iloc[start:end]
        output_file = input_file.replace('.tsv', f'_subpart{i + 1}.tsv')
        df_part.to_csv(output_file, sep='\t', index=False, header=False)

# usage example #
# if __name__ == '__main__':
# to just clean the sequences and save them in a new file
# clean_vdjdb_seqs("/path/to/vdjdb_downloaded/cmv_23072024.tsv")
# to clean and split the sequences into 3 parts
# clean_vdjdb_seqs("/path/to/vdjdb_downloaded/cmv_23072024.tsv", split_df=True)
# to split a file into n parts
# split_file_n_parts("/path/to/vdjdb_downloaded/cmv_23072024_part1.tsv", 3)
