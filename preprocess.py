import random
# import sys
import pandas as pd

with open("dev.de", "r") as ins:
    dev_de = []
    for line in ins:
        dev_de.append(line.strip('\n'))

with open("dev.en", "r") as ins:
    dev_en = []
    for line in ins:
        dev_en.append(line.strip('\n'))

with open("dev_generated.txt", "r") as ins:
    devg_en = []
    for line in ins:
        devg_en.append(line.strip('\n'))

src_lines = dev_de + dev_de
        
lines = []
labels = []
gid = 0
guids = []
all_guids = [i for i in range(len(2*devg_en))]

for line in dev_en:
  lines.append(line)
  labels.append('human')
  gind = random.randint(0,len(all_guids) - 1)
  gid = all_guids[gind]
  del all_guids[gind]
  guids.append(gid)

for line in devg_en:
  lines.append(line)
  labels.append('machine')
  gind = random.randint(0,len(all_guids) - 1)
  gid = all_guids[gind]
  del all_guids[gind]
  guids.append(gid)
print(len(src_lines), len(lines))
assert(len(src_lines) == len(lines))
# dev_data = {'guid' : guids,'text_a': src_lines ,'text_b': lines,'label':labels}
dev_data = {'guid' : guids,'text_a': lines,'label':labels}
df_train = pd.DataFrame(dev_data, columns=["guid", "text_a","label"])
df_train = df_train.sort_values(by=['guid'])
df_train.to_csv("dev_bert_src2.csv", index=False)