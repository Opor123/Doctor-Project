import pandas as pd

tumor=pd.read_csv('data\\tumor.csv')
breast=pd.read_csv('data\Breast_cancer.csv')

tumor['Class']=tumor['Class'].map({2:0,4:1})

tumor.rename(columns={'Class': 'diagnosis'},inplace=True)

tumor.drop(columns=['Sample code number'],inplace=True)

tumor.to_csv('tumor_cleaned.csv',index=False)

print('Breast Cancer column: ',breast.columns)
print('Tumor data column: ',tumor.columns)

merge=pd.concat([breast,tumor],ignore_index=True)

merge.to_csv('Merge_data.csv',index=False)

