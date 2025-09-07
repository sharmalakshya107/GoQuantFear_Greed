import pandas as pd
from utils.tagging import detect_tags

# Load the old sentiment history
infile = 'data/sentiment_history.csv'
outfile = 'data/sentiment_history_retagged.csv'
df = pd.read_csv(infile)

def retag(row):
    tags = detect_tags(str(row['tags']) if pd.notnull(row['tags']) else '')
    return '|'.join(tags)

# Re-tag every row using the new logic
print('Re-tagging sentiment history...')
df['tags'] = df.apply(lambda row: retag(row), axis=1)

df.to_csv(outfile, index=False)
print(f'Retagged sentiment history written to {outfile}') 