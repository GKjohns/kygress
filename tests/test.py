import pandas as pd
from kygress import regress


df = pd.read_pickle('./tests/test_data/df_test.pkl')
X = ['is_gif', 'post_age_inverse', 'title_is_question']
y = ['score_log', 'comment_count_log', 'upvote_ratio']

df_results = regress(df, X, y, fdr_correction=)

print(df_results.sample(3).T)

