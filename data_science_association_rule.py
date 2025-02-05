import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("retail_dataset.csv")

# one_hot_encoding_obj = OneHotEncoder(sparse_output=False)
# final_hot_coded_data = one_hot_encoding_obj.fit_transform(df[["1","2","3","4","5"]])
# print(final_hot_coded_data)

items = df['0'].unique()
item_count_data =df['0'].nunique()
print(items)
print(item_count_data)

final_list = []

for index, row in df.iterrows():
    my_dict = {}
    uncommons = list(set(items) - set(row))
    print(uncommons)
    commons = list(set(items).intersection(row))
    print(commons)
    for uc in uncommons:
        my_dict[uc] = 0

    for com in commons:
        my_dict[com] = 1
    final_list.append(my_dict)

print(final_list)
print(len(final_list))

final_data_frame = pd.DataFrame(final_list)
freq_items = apriori(final_data_frame,min_support=0.2,use_colnames=True,verbose=1)
print(freq_items.head())

final_association = association_rules(freq_items,metric="confidence",min_threshold=0.6,num_itemsets=True)
print(final_association)

final_association.to_csv("final_output.csv")
