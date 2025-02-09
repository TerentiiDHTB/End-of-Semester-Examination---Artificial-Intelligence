import pandas as pd

url_regexp = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
number_regexp = r"^[-+]?[0-9]+$"

def remove_junk_from_summary(summary):
  summary = summary.lower()

  summary = re.sub(url_regexp, '', summary)
  summary = re.sub(number_regexp, '', summary)

  return summary

def serialiseScore(score):
    return 1 if score > 0 else 0

df = pd.read_csv("ai-model/data.tsv", sep="\t")

# print(df.head())

df.drop(["link", "published", "tickers"], axis=1, inplace=True)

# print(df.info())

#pd.set_option('display.max_colwidth', None)

df[df['summary'].isnull()]['title']
#as I can see, 106, 150, 253, 342 news makes no sense without summary. So I can transfer titles of other news to summary column and delete title column.

ids = [20, 139, 213, 272, 303, 319, 334, 490]

df.loc[ids, 'summary'] = df.loc[ids, 'title']

# df.info()

df.drop('title', axis=1, inplace=True)

df.dropna(inplace=True)

df['summary'] = df['summary'].apply(remove_junk_from_summary)
df['score'] = df['score'].apply(serialiseScore)


