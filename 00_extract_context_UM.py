import pandas as pd
import numpy as np

data = ('/dataset/')
output = ('/compiled_cleaned/')

bi_edu = pd.read_csv(data + 'Bi-Education.txt', sep = '\t', names = ['sentence'])
bi_edu['UM_source'] = 'Education'
bi_edu

# do the same for all the other registers

bi_dfs = pd.concat([bi_edu, bi_law, bi_mic, bi_new, 
	bi_sci, bi_spo, bi_sub, bi_the, bi_tes], 
	axis = 0, join = 'outer', ignore_index=True)
bi_dfs

# only select odd rows
bi_dfs_ch = bi_dfs.iloc[1::2]  
bi_dfs_ch = bi_dfs_ch.reset_index(drop = True)
bi_dfs_ch['sentence'][0]
len(bi_dfs_ch['sentence'][0])

# exclude sentences that are too short or too long
bi_dfs_ch_len = bi_dfs_ch[(bi_dfs_ch['sentence'].str.len() > 4) & (bi_dfs_ch['sentence'].str.len() < 21)]
bi_dfs_ch_len = bi_dfs_ch_len.reset_index(drop = True)
bi_dfs_ch_len

sent_dict = list(bi_dfs_ch_len['sentence'])
def extract_sentence_containing_word(txt, target_word):
  if type(target_word) != float:
    if len(target_word) > 0:
      result = [sentence for sentence in txt if target_word in sentence]
    return result
  else:
    return np.nan
df_tsi = pd.read_csv(data + 'COS960_all.txt', sep = ' ')
df_tsi = df_tsi.iloc[0:959, 0:3]
df_tsi.columns = ['word1', 'word2', 'average_human_ratings']
df_tsi

df_tsi['word1_sentences'] = df_tsi['word1'].apply(lambda x: extract_sentence_containing_word(sent_dict, x))
df_tsi['word1_sentences_count'] = df_tsi['word1_sentences'].str.len()
df_tsi['word2_sentences'] = df_tsi['word2'].apply(lambda x: extract_sentence_containing_word(sent_dict, x))
df_tsi['word2_sentences_count'] = df_tsi['word2_sentences'].str.len()
df_tsi

# then do the same for all the other datasets


