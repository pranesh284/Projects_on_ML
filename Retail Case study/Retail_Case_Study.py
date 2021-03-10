

# ## Q1. Preparing Dataset

# In[83]:


import pandas as pd


# In[3]:


Customer = pd.read_csv("Customer.csv")
Customer


# In[4]:


Product_hierarchy = pd.read_csv("prod_cat_info.csv")
Product_hierarchy


# In[5]:


Transaction = pd.read_csv("Transactions.csv")
Transaction


# In[6]:


# 1. Merging
Customer_Trans = pd.merge(left = Customer,
                          right = Transaction,
                          left_on = 'customer_Id',
                          right_on = 'cust_id',
                          how = 'inner',
                          indicator = True)


# In[7]:


Customer_Trans


# In[99]:


Customer_Final = pd.merge(left = Customer_Trans,
                          right = Product_hierarchy,
                          left_on = 'prod_cat_code',
                          right_on = 'prod_cat_code',
                          how = 'inner'
                          )



# In[100]:


Customer_Final


# ## Q2. Summary Report

#
# ### Column names with their data-types
#

# In[11]:


Customer_Final.dtypes


# ### Top 10 Observations

# In[12]:


Customer_Final.head(10)


# ### 10 Bottom Observations

# In[13]:


Customer_Final.tail(10)


# ### Five Number Summary

# In[14]:


import numpy as np
Data_min = Customer_Final['total_amt'].min()
Data_max = Customer_Final['total_amt'].max()
Data_q1  = np.percentile(Customer_Final.total_amt,25)
median  = np.percentile(Customer_Final.total_amt,50)
Data_q3  = np.percentile(Customer_Final.total_amt,75)
print('Min = ',Data_min)
print('Max = ',Data_max)
print('Median = ',median)
print('Q1 = ',Data_q1)
print('Q3 = ',Data_q3)


# ## Frequency Table :
#
# ### Store type

# In[15]:


freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['Store_type'])
freq_table.columns = ['TeleShop','MBR','e-shop','Flagshipstore']
freq_table.index = ['Male','Female']
freq_table


# ### Prod_cat

# In[16]:


freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['prod_cat'])

freq_table.columns = ['Books','Bags','Clothing','Footwear','Electronics','Home and kitchen']
freq_table.index = ['Male','Female']
freq_table


# ### Prod_subcat

# In[17]:


freq_table = pd.crosstab(index = Customer_Final['Gender'],
                         columns = Customer_Final['prod_subcat'])
freq_table.columns = ['Men','Women','Kid','Mobile','Computer','Personal Appliances','Cameras','Audio and video',
                      'Fiction','Academic','Non-fiction','Children','Comics','DIY','Furnishing','Kitchen',
                      'Bath','Tools']
freq_table.index = ['Male','Female']
freq_table


# ## Q3. Histograms for all continuous variables and frequency bars for categorical variables

# ### Histogram for continous variables -
#
#
# ### 1. Tax

# In[20]:


import matplotlib.pyplot as plt
Tax = Customer_Final['Tax']
plt.hist(Tax,color=['yellow'])
plt.xlabel('tax')
plt.ylabel('Frequency')
plt.show()


# ### 2. Total amount

# In[19]:


Total_Amt = Customer_Final['total_amt']
plt.hist(Total_Amt,color = 'Blue')
plt.xlabel('Total amount')
plt.ylabel('Frequency')
plt.show()


# ### Frequency Bar for Categorical variables -
#
#
#
# ### 1. Gender

# In[21]:


Customer_Final['Gender'].value_counts().plot(kind = 'bar')


# ### 2. Store type

# In[22]:


Customer_Final['Store_type'].value_counts().plot(kind = 'bar')


# ### 3. Product category

# In[23]:


Customer_Final['prod_cat'].value_counts().plot(kind = 'bar')


# ### 4. Product sub category

# In[24]:


Customer_Final['prod_subcat'].value_counts().plot(kind = 'bar')


# ## Q4
#
# ### A. Time period of the available transaction data

# In[ ]:





# ### B. Count number of negative total amount

# In[25]:


df = Customer_Final['total_amt']
count2 = Customer_Final.loc[(df<0),['total_amt']].count()
count2


# ## Q5. Analyze which product categories are more popular among females vs male customers.

# In[134]:


# Popular among Male
M = Customer_Final.loc[Customer_Final['Gender']=='M']

group_prod = M.groupby(['prod_cat'])['total_amt'].sum()
popular_M = group_prod.nlargest(1)
display('The most popular product category in Male customers is : ',popular_M)

# Popular among Female
F = Customer_Final.loc[Customer_Final['Gender']=='F']
group_prod1 = F.groupby(['prod_cat'])['total_amt'].sum()
popular_F = group_prod1.nlargest(1)
display('The most popular product category in Female customers is : ',popular_F)


# #### Among Male vs Female the most popular product category is Books.

# ## Q6. Which City code has the maximum customers and what was the percentage of customers from that city?

# In[173]:


max_cust = Customer['city_code'].value_counts()
t = max_cust.nlargest(1)

display("City code which has Maximum customers is : ",t)

#percentage of customers from city code 3
tot_customer = Customer['customer_Id'].count()
percent = round((595/tot_customer)*100,2)
print("Percentage of customers from the city code 3 is {}% : ".format(percent))


# ## Q7. Which store type sells the maximum products by value and by quantity?

# In[24]:


sort_list = Customer_Final.sort_values(['total_amt','Qty'],ascending = False)
display(sort_list.head(1)['Store_type'])


# ## Q8. What was the total amount earned from the "Electronics" and "Clothing" categories from
#Flagship Stores?

# In[32]:


df = pd.DataFrame(Customer_Final)
tf = df[df.prod_cat.isin(['Electronics','Clothing']) & (df.Store_type == 'Flagship store')]
total = tf.total_amt.sum()
print('Total amount earned',total)


# ## Q9. What was the total amount earned from "Male" customers under the "Electronics" category?

# In[52]:


tf1 = df[(df.Gender == 'M') & (df.prod_cat == 'Electronics')]
total = tf1.total_amt.sum()
print('Total amount earned',total)


# ## Q10. How many customers have more than 10 unique transactions, after removing all transactions which have any negative amounts?

# In[46]:


df1 = df[(df.total_amt > 0)]
ts = df1.transaction_id.nunique()
print('Total customers having more than 10 unique transactions are - ',ts)


# ## Q11. For all customers aged between 25 - 35, find out:
#
#
# ### a. What was the total amount spent for “Electronics” and “Books” product categories?

# In[47]:


curr_year = pd.to_datetime('today').year
dob_year = pd.DatetimeIndex(df['DOB']).year          #extract year from DOB

x = dob_year-100                                               # for the years which belongs to 60's
v = curr_year - x
y = curr_year - dob_year
df['age'] = (np.where(dob_year > curr_year,v,y))
df


# In[174]:


total = df.loc[((df.age >25) & (df.age <35)) & ((df.prod_cat=='Books') | (df.prod_cat=='Electronics'))]['total_amt'].sum()
print('Total amount spent',total)


# ### b.  What was the total amount spent by these customers between 1st Jan, 2014 to 1st Mar, 2014?

# In[92]:


Customer_Final['tran_date'] = pd.to_datetime(Customer_Final['tran_date'])

t_date = Customer_Final[(Customer_Final['tran_date'] > '2014-01-01') & (Customer_Final['tran_date'] < '2014-03-01')]
total_amount = t_date.total_amt.sum()
print('Total amount spent by the customer - ',total_amount)


