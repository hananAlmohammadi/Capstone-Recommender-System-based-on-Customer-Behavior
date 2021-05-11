#--------------------------------importing libraries-------------------------------------------------------------------#

import pickle 
import random
import numpy as np 
import pandas as pd
from random import sample

#--------------------------------Reading files-------------------------------------------------------------------#

# Opening cleand dataset file 
def get_cleand_df(path):
     with open(path, 'rb') as infile:
         df=pickle.load(infile)
         return df

# Opening predicted dataset file 
def df_after_pre(path):
     with open(path, 'rb') as infile:
         df=pickle.load(infile)
         return df
         
# Opening corrlation matrix from SVD model 
def get_corr_mat(path):
     with open(path, 'rb') as infile:
         df=pickle.load(infile)
         return df

# Opening product columns from pivot table for the event history of products
def get_event(path):
     with open(path, 'rb') as infile:
         df=pickle.load(infile)
         return df

path1 ='C:/Users/hano0/Desktop/DSI8/capstone/DataArmors/'
# path='C:/Users/hano0/Desktop/DSI8/atom/shopping.pkl'
##df1=read_df(path)
newdf= df_after_pre(path1 + 'prediction.pkl')
corr_mat= get_corr_mat(path1+'corrmat.pkl') 
event_product= get_event(path1+'event_column.pkl')
cleand_data = get_cleand_df(path1+'cleand_data.pkl')
product_list = list(event_product)

#------------------------------------------------------------------------------------------------------------------#

# This finction takes list of products that similar to the product we want a recommendation for it
# and return a product that has minimum price with its brand...
def similar_product_with_less_price(corr_pr1 , data1):
    list_of_product=list(event_product[(corr_pr1<1.0) & (corr_pr1 > 0.7)]) 
    pr_ids=[]
    list_brand=[]
    list_price=[]
    #The for will iterate through each product and 
    # save its price, IDs and brand in seperate lists
    for i in list_of_product:
        df_pr = data1[data1['product_id']== i].iloc[0]  # return a row that match the produect_id 
        brand=df_pr['brand']
        price=df_pr['price']
        list_brand.append(brand)
        list_price.append(price)
        pr_ids.append(i)
    # gitting the minimum price and use its index to get its ID and brand..
    if len(list_price)==0:
        min1=0
        min_idx=0
        prid_min  = 0000
        brand_min = '0'
    else:
        min1 = min(list_price)
        min_idx=list_price.index(min1)
        prid_min  = pr_ids[min_idx]
        brand_min = list_brand[min_idx]
    return prid_min , min1 , brand_min

#------------------------------------------------------------------------------------------------------------------#

#This function will recommend similar product for a specific product ..
def Get_Recommendation(prid , data1 , newdf  ,product_list ,corr_mat): 

  #the for loop will iterate through each product in the predcted dataset to check predicted label(purchase)=0
  # it will send it to "similar_product_with_less_price" function to find product witl less price..
  for  x ,i  in  enumerate(newdf['product']):
    newdf[['product']].iloc[[x]].index.values.astype(int)[0] 
    if prid in i:
      row_index = x
      if newdf['purchase_pred'].iloc[row_index] == 0:  
          pr1 = product_list.index(prid)
          corr_pr1 = corr_mat[pr1] 
          pr2 , pmin , brand= similar_product_with_less_price(corr_pr1, data1)
  return pr2 , pmin ,brand

#-------------------------------------------Prining brand list---------------------------------------------------#

def brand_list():
    brand = pd.DataFrame(cleand_data['brand'])
    unique_products=pd.DataFrame(brand['brand'].unique())
    random_products1= unique_products.sample(10)
    random_products1.rename(columns={0: 'brand'}, inplace=True)
    return list(random_products1['brand'])

    #-------------------------------------Taking Input from the user --------------------------------------------#

def get_id(brand_input):
    
   merged_df=pd.merge(cleand_data,newdf,on='user_session')
   random_product_df = merged_df[merged_df['brand'] == brand_input][['product_id','user_session']]
   random_product=[]  
   for i in random_product_df.product_id:
      if i in event_product:
         random_product.append(i)
   random_id=random.choice(random_product)
   pr2 , pmin , brand=Get_Recommendation(random_id , cleand_data , newdf , product_list ,corr_mat)
   return pr2 , pmin , brand
    

