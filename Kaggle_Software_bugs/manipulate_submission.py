import pandas as pd

df = pd.read_csv('./submissions/submissions.csv',names=['id','drop_column','defects'])

print(df.head())

df = df.drop(['drop_column'],axis=1)

df.to_csv("./submissions/final_submission.csv",index=False)