#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv('Downloads\Credit card transactions - India - Simple.csv')


# In[4]:


df.head()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv("Downloads\Credit card transactions - India - Simple.csv")
df


# In[8]:


df.head(10)


# In[9]:


df.columns


# In[10]:


df.info()
# we can see that there only one numeric column


# In[11]:


df.nunique()


# In[12]:


df.describe()


# In[13]:


df.isnull().sum()
# there is no Null Values


# In[14]:


df.set_index('index', inplace = True)
df


# In[15]:


# change the dtype of Date column to date time 
df['Date'] = pd.to_datetime(df['Date'])
df


# In[16]:


# make the data more understandable
# change F to (Female) amd M to (Male)

df['Gender'] = df['Gender'].replace('F', 'Female')
df['Gender'] = df['Gender'].replace('M', 'Male')
df


# In[17]:


# split the city column to city and country columns

df[['city', 'country']] = df['City'].str.split(',', expand=True)
df


# In[18]:


# Grouping by 'Card Type' and calculating the mean amount spent for each card type
card_type = df.groupby('Card Type')['Amount'].mean()
print("Mean Amount Spent by Card Type:")
print(card_type)


# Grouping by 'Exp Type' and calculating the total amount spent in each expense category
expense_type = df.groupby('Exp Type')['Amount'].sum()
print("\nTotal Amount Spent by Expense Type:")
print(expense_type)


# Grouping by 'Gender' and calculating the total amount spent by each gender
gender = df.groupby('Gender')['Amount'].sum()
print("\nTotal Amount Spent by Gender:")
print(gender)


# Grouping by 'City' and calculating the total amount spent in each city
city = df.groupby('city')['Amount'].sum()
print("\nTotal Amount Spent in each City:")
print(city)


# Grouping by 'City' and 'Card Type' and calculating the mean amount spent for each combination
city_card = df.groupby(['city', 'Card Type'])['Amount'].mean()
print("\nMean Amount Spent by City and Card Type:")
print(city_card)


# In[19]:


# mean of transactions depends on Card Type
c_type = df.groupby(df['Card Type']).mean()
c_type = c_type.sort_values(by = 'Amount')
c_type


# In[20]:


# line plot for mean of transactions depends on Card Type

plt.style.use('fivethirtyeight')

plt.plot(c_type.index, c_type, marker = 'o')

plt.xlabel('Card Type')
plt.ylabel('Amount')
plt.title('Mean of Transactions by Card Type')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability if needed
plt.tight_layout()  # Adjust the layout to prevent clipping of labels
plt.show()


# In[21]:


# bar chart for card types number of occurrences 

# Count the occurrences of each card type
card_type_counts = df['Card Type'].value_counts()


plt.figure(figsize=(8, 6))
plt.bar(card_type_counts.index, card_type_counts.values)
plt.xlabel('Card Type')
plt.ylabel('Count')
plt.title('Distribution of Card Types')
plt.xticks(rotation=45)
plt.show()


# In[22]:


# Plot Stacked Bar Chart of Card Types by Gender

# Cross-tabulate Card Type and Gender
ct = pd.crosstab(df['Card Type'], df['Gender'])

plt.figure(figsize=(10, 8))
ax = ct.plot(kind='bar', stacked=True, edgecolor='k')

# Add the count values on top of each segment of the bars
for i in ax.patches:
    ax.text(i.get_x() + i.get_width() / 2, i.get_y() + i.get_height() / 2, str(i.get_height()), 
            fontsize=10, color='white', ha='center', va='center')

plt.xlabel('Card Type')
plt.ylabel('Count')
plt.title('Card Types by Gender')
plt.xticks(rotation=45)
plt.legend(title='Gender', loc='upper right')
plt.show()


# In[23]:


grouped_by_gender = df.groupby('Gender')['Amount'].sum()

# Displaying the total amount spent by each gender
print("Total Amount Spent by Gender:")
print(grouped_by_gender)

# Plotting the gender spending comparison as a bar plot
plt.figure(figsize=(6, 4))
grouped_by_gender.plot(kind='bar', edgecolor='k')
plt.xlabel('Gender')
plt.ylabel('Total Amount Spent')
plt.title('Gender Spending Comparison')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()


# In[24]:


# mean of transactions depends on Exp Type
exp_type = df.groupby('Exp Type')['Amount'].mean()
exp_type.sort_values()
exp_type


# In[25]:


# scatter plot for mean of transactions depends on expensies Type

plt.style.use('fivethirtyeight')

plt.scatter(exp_type.index, exp_type, marker = 'o', s = exp_type/450)

plt.xlabel('Expenses Type')
plt.ylabel('Amount')
plt.title('Mean of Transactions by Expenses Type')
plt.xticks(rotation=0)  # Rotate x-axis labels for readability if needed
plt.tight_layout()  # Adjust the layout to prevent clipping of labels
plt.show()


# In[26]:


# bar plot for top spending Expense Category

grouped_by_expense_type = df.groupby('Exp Type')['Amount'].sum()
top_spending_categories = grouped_by_expense_type.sort_values(ascending=False)

# Displaying the top spending categories
print("Top Spending Categories:")
print(top_spending_categories)

# Plotting the top spending categories as a bar plot
plt.figure(figsize=(10, 6))
top_spending_categories.plot(kind='bar', edgecolor='k')
plt.xlabel('Expense Category')
plt.ylabel('Total Amount Spent')
plt.title('Top Spending Categories')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[27]:


# pie chart for card types ( percent of occurrences )

# Count the occurrences of each expense type
expense_type_counts = df['Exp Type'].value_counts()


plt.figure(figsize=(8, 8))
plt.pie(expense_type_counts.values, labels=expense_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Proportion of Expense Types')
plt.show()


# In[28]:


# Bar plot for Expense Type (Exp Type)
plt.figure(figsize=(8, 6))
expense_type_counts = df['Exp Type'].value_counts()
expense_type_counts.plot(kind='bar', edgecolor='k')

# Add the values on top of each bar
for index, value in enumerate(expense_type_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')

plt.xlabel('Expense Type')
plt.ylabel('Frequency')
plt.title('Distribution of Expense Type')
plt.show()


# Bar plot for Gender
plt.figure(figsize=(8, 6))
gender_counts = df['Gender'].value_counts()
gender_counts.plot(kind='bar', edgecolor='k')

# Add the values on top of each bar
for index, value in enumerate(gender_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')

plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Distribution of Gender')
plt.show()


# In[29]:


# Haighest and lowest 10 cities in transctions by mean

high_city= df.groupby('city')['Amount'].mean().sort_values(ascending = False).head(10)
h_size = high_city
high_city

low_city= df.groupby('city')['Amount'].mean().sort_values().head(10)
l_size = low_city
low_city


# In[30]:


# Haighest and lowest 10 cities in transctions by mean plot

plt.figure(figsize= (31, 8))


plt.subplot(131)
plt.scatter(high_city.index, high_city, label = 'Haighest' , s = h_size/250, c = 'blue', edgecolors = 'black')
plt.xlabel("City")
plt.ylabel("Amount of Transction")
plt.title('Mean of Haighest cities in transctions')
plt.legend(loc='upper right')


plt.subplot(132)
plt.scatter(low_city.index, low_city, label = 'lowest' , s = l_size/250, c = 'red', edgecolors = 'black')
plt.xlabel("City")
plt.ylabel("Amount of Transction")
plt.title('Mean of lowest cities in transctions')
plt.legend(loc='upper left')


# plt.rcParams['figure.figsize'] = ()

plt.show()


# In[31]:


# Plot the histogram Distribution of Transaction Amount
plt.figure(figsize=(8, 6))
n_bins = 10
plt.hist(df['Amount'], bins=n_bins, edgecolor='black')

plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Transaction Amount')
plt.show()


# In[32]:


# line chart for total amount spent each Day

daily_spending = df.groupby('Date')['Amount'].sum()

plt.figure(figsize=(10, 6))
plt.plot(daily_spending.index, daily_spending.values)
plt.xlabel('Date')
plt.ylabel('Total Amount Spent')
plt.title('Daily Total Amount Spent Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[33]:


# line chart for total amount spent each Month
monthly_spending = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()

# Convert the PeriodIndex to string format for plotting
monthly_spending.index = monthly_spending.index.astype(str)

plt.figure(figsize=(10, 6))
plt.plot(monthly_spending.index, monthly_spending.values, marker='o')
plt.xlabel('Month')
plt.ylabel('Total Amount Spent')
plt.title('Monthly Total Amount Spent Over Time')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[35]:


# line chart for total amount spent each year

yearly_spending = df.groupby(df['Date'].dt.year)['Amount'].sum()

plt.figure(figsize=(10, 6))
plt.plot(yearly_spending.index, yearly_spending.values, marker='o')
plt.xlabel('Year')
plt.ylabel('Total Amount Spent')
plt.title('Yearly Total Amount Spent Over Time')
plt.xticks(yearly_spending.index, rotation=45)
plt.grid(True)
plt.show()


# In[36]:


# Stacked Area Chart of Expense Types over Time

# Pivot the data to get the expense types as columns and sum the amounts for each date
pivot_df = df.pivot_table(index='Date', columns='Exp Type', values='Amount', aggfunc='sum', fill_value=0)

plt.figure(figsize=(10, 6))
plt.stackplot(pivot_df.index, pivot_df.values.T, labels=pivot_df.columns)
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.title('Stacked Area Chart of Expense Types over Time')
plt.xticks(rotation=30)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()

