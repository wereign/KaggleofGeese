from autogluon.tabular import TabularDataset, TabularPredictor

train_df= TabularDataset('./data/train.csv')

print()
print(train_df.head())
print()

label = 'defects'
train_df[label].describe()

predictor = TabularPredictor(label=label).fit(train_df,ag_args_fit={'num_gpus': 1})
test_data = TabularDataset('./data/train.csv')
y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

predictor.evaluate(test_data, silent=True)

predictor.save()