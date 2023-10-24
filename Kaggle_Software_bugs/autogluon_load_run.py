from autogluon.tabular import TabularDataset, TabularPredictor


test_data = TabularDataset('./data/test.csv')

predictor = TabularPredictor.load(r"./AutogluonModels/ag-20231023_154507/")


y_pred = predictor.predict_proba(test_data)



y_pred['id'] = test_data['id']
print(y_pred.head())
print(y_pred.columns)
y_pred = y_pred[['id',False,True]]

y_pred.to_csv('./submissions/submissions.csv',index=False)
