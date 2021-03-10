
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('http://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
df.head()
df.describe()
df.shape

df.to_csv("online_store.csv")




df = pd.read_csv('online_store.csv')



#remove extra spaces from descriptions
#remove rows with empty invoices and containing C
df['Description'].head()

df['Description'] = df['Description'].str.strip()
df['Description'].head()

df.InvoiceNo.head()

df.dropna(axis=0, subset=['InvoiceNo'], inplace=True) #remove NA invoice nos

df.dtypes

df['InvoiceNo'] = df['InvoiceNo'].astype('str')



df = df[~df['InvoiceNo'].str.contains('C')]
df


pd.set_option('display.max_columns', None)
df
basket = (df[df['Country'] =="United Kingdom"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


basket

basket.shape[0] * basket.shape[1]

basket.to_csv("basket.csv")


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)

basket_sets.head()

basket_sets.columns


basket_sets.drop('POSTAGE', inplace=True, axis=1) #remove POSTAGE column not an item



frequent_itemsets = apriori(basket_sets, min_support=0.02,  use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="lift")
rules[['antecedents','consequents', 'support', 'confidence', 'lift']].to_csv("Basket1.csv")


rules[ (rules['lift'] >= 6) &  (rules['confidence'] >= 0.8) ]

basket['ALARM CLOCK BAKELIKE GREEN'].sum()

basket['ALARM CLOCK BAKELIKE RED'].sum()


basket2 = (df[df['Country'] =="Germany"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket2



basket_sets2 = basket2.applymap(encode_units)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)

rules2[ (rules2['lift'] >= 4) &   (rules2['confidence'] >= 0.5)]

