# Blending csvs at selected folder.

import os
import glob
import pandas as pd
import random

os.chdir("C:\\Users\\Semih\\Desktop\\New folder")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

df = pd.read_csv(all_filenames[0])
for f in all_filenames[1:]:
     other = pd.read_csv(f)
     df = df.join(other.set_index('SK_ID_CURR'), on='SK_ID_CURR',lsuffix = all_filenames.index(f), rsuffix = all_filenames.index(f))

sk_col = ['SK_ID_CURR']
target_cols = [f"TARGET{str(count)}" for count, value in enumerate(df.columns) if "TAR" in value]
print(target_cols)
sk_col.extend(target_cols)

df.columns = sk_col

randomlist = random.sample(range(1, 200), len(target_cols))
df['TARGET'] = 0

for count, value in enumerate(randomlist, 1):
     df['TARGET'] += value * df['TARGET'+str(count)]
     #print(df)
df['TARGET'] = df['TARGET'] / sum(randomlist)


df = df[['SK_ID_CURR', 'TARGET']]
print(df)
print(randomlist)
df.to_csv( f"combined_csv{randomlist}.csv", index=False, encoding='utf-8-sig')

