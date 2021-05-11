
![](https://camo.githubusercontent.com/1a91b05b8f4d44b5bbfb83abac2b0996d8e26c92/687474703a2f2f692e696d6775722e636f6d2f6b6538555354712e706e67)
![](https://i.ibb.co/KmXhJbm/Webp-net-resizeimage-1.png)
![](https://www.citc.gov.sa/_catalogs/masterpage/CITC_New_Design/assets/images/logos/MCIT_logo.png)

# Capstone Project: Recommender System for Products based on Customer Behavior.![]([https://files.slack.com/files-pri/T0351JZQ0-F01M87K490X/kptwcmckyjuwgbdbrzr9mx.png](https://files.slack.com/files-pri/T0351JZQ0-F01M87K490X/kptwcmckyjuwgbdbrzr9mx.png)) 

### Install

Project is created with:

  

![Python](https://img.shields.io/badge/Python-3.7-blueviolet)

![Pandas](https://img.shields.io/badge/Pandas-1.1.5-pink)

![Numpy](https://img.shields.io/badge/Numpy%20-1.19.5-blue)

![scipy](https://img.shields.io/badge/scipy-1.6.0-violet)

![anaconda](https://img.shields.io/badge/Anaconda-Navigator-green)

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.4.1-yellow)

![Keras](https://img.shields.io/badge/Keras%20-2.4.3-red)
  
![Framework](https://img.shields.io/badge/Framework-Flask-navy)

![Frontend](https://img.shields.io/badge/Frontend-HTML/CSS-orange)



# Introduction 
Previously, E-commerce websites have been faced problems in understanding user preferences. Nowadays, recommendation systems solve this issue in various ways and play an essential role in e-commerce and most companies use it to increase their profit. They are software tools that adopted data mining and machine learning techniques and represent user preferences. The recommendation systems aiming to satisfies the customers by recommending products to purchase or examine them based on their behavior patterns. There are different techniques that have been used for applying recommendation systems, including content-based, collaborative, knowledge-based and etc... These techniques sometimes have been combined into the hybrid model to improve the recommender performance. In this project, a collaborative-based recommender will be implemented along with a predictive model that adopted sequential recurrent neural networks (RNN) machine-learning algorithm to enhance the recommender system performance.



## Problem Statment
Today, there is a huge volume of products available and sometimes customers get confused about what product to choose, here comes the importance of the recommender system which will narrow the options and guide them towards products similar to what they like previously and they will be more confident to buy the products. In this project, we aiming to build a collaborative, item-based recommender system based on customer behavior patterns and a predictive model that implemented using sequential recurrent neural networks (RNN) machine learning algorithms (simplRNN and Long short-term memory LSTM) in order to enhance the recommender system performance. Along with that, we will use a singular value decomposition (SVD) as a dimensionality reduction technique to extract features and measure correlation from the item-item matrix that will be used in finding the similarity between products.

## Get Started
1- Install Git bash terminal

2- Clone this repository to your local machine using bash terminal:

	> git clone (repository link)
 
3- Install conda and create an environment:

	> conda create -n (environment name)
	
4- Activate your enviroment:

    > conda/source activate (environment name)
	
5- Install all the libraries in requirements.txt file as follows:

	> pip install -r requirements.txt

6- Open your terminal from your project folder and run the following command to excute the file app.py:

 	> python app.py
 
7- Go to your browser and type http://127.0.0.1:5000/ in the address bar.

8- If you want to edit on  the code you need to download VS code and open the project folder in it.
## Datasets Description
The datasets come from an online store called Macy's, the first one called "shopping". It contains information about cosmetics products such as the product's price, its brand, the customers like the user sessions and his ID, and the events such as the type of the event (view, cart, remove, purchase) and each event are time tagged. The second dataset called "session_ features" and we used it only in the EDA section. Basically, the difference is one hot encoded for the following features 
event_time, event_type, year, month, weekday, and hour. These datasets include all customer behavior. Each dataset contains 1500000 rows, 36 features in session_features, and 12 features in the shopping dataset.

|Feature|Type|Dataset|Description|  
|---|---|---|---|  
|event_time|datetime64|shopping|Contains the date and time for event|  
|event_type|object|shopping|Contains the type of event like purchase, cart, view, remove from cart |  
|product_id|int64|shopping|Contains the ID for each product|  
|category_id|int64|shopping|Contains the ID for product category|  
|brand|object|shopping|Contains the brand for each product|  
|price|float|shopping|Contains the price for each product|  
|user_id|int64|shopping|Contains the ID for each user|  
|user_session|object|shopping|Contains the session of user|  
|year|int64|shopping|Contains the year of event based on session|  
|month|int64|shopping|Contains the month of event based on session|  
|weekday|object|shopping|Contains the weekday of event based on session|  
|hour|int64|shopping|Contains the hour of event based on session|


 ## EDA
In this section, we provide plots that visulize the dataset in order to give better understanding about the dataset.
 #### Heatmap  
<img src="/images/heatmap.PNG" width="600">
- The figure above shows us the correlation between features.



#### The distrbution of events during the week  
<img src="/images/weekday with event.PNG" width="600">
- The figure above shows us the interaction time of events on the first days of the week higher than the rest of the week. in our opinion people usually at the weekend spend their time with friends and family so, they maybe don't have time to shop online.



#### The percentage of event type in the website  
<img src="/images/percntage.PNG" width="600">
- The figure above shows us the percentage per event type on the website, we notice the view event gets 68.6%, which's a higher percentage than purchase which is 3.43% of customers buy products. So, most people view products and they didn't purchase.



####  Distribution  of purchased products price  
<img src="/images/pricedis.PNG" width="600">
-The figure above shows us the distribution of the purchased products price, we notice it's left-skewed, and most of the values between  0 to 150. 



#### Relationship between interaction time of user with purchase event  
<img src="/images/eventvspurchaseevent.PNG" width="600">
- The figure above shows us that most of the customers who spend more time on the website just a few of them buy products. And the customers who actually buy a product spend less time on the website.



#### Relationship between day time preiod and purchase event  
<img src="/images/time-purchase-event.PNG" width="600">
- The figure above shows that most customers spend their time on the website during the afternoon and dawn whether they purchase or not.


#### Relationship between interaction time and purchase event per weekday  
<img src="/images/perweekday.PNG" width="600">
- The figure above shows that customers who didn't purchase any product spend most of their time on the website on Friday and customers usually purchase spend most of their time on the website on Saturday and Thursday. 

#### Distrbution of browsing time per hour
<img src="/images/browsing.PNG" width="600">

- The above plot shows the brwsing time per houre, we notice that the traffic in the website is higher at 19 PM and at 1 AM will be the lowest traffic in the website.

-----------------------------------------------------------
## Recommender System Architecture   
<img src="/images/Recommender System Architecture.png" width="600">

## The Recommender System Development  
Our system composed of three main stages:  
- The predictive model  
- The recommender system  
- The web app
### Predictive  Model  
 The first stage is the predictive model that will analyze the sequence of customer's past behavior to detect patterns in the data. There are various types of customers actions in a website which contain a sequence of events (view, cart, remove, purchase) that are generated per session and it can predict if a purchase would occur by the end of the session or not. The model implemented by many to one SimpleRNN deep machine learning model since RNN is specialized to deals with the sequence of data. 

### Recommender System  
The second stage is the recommender system we used the item-based approach for the collaborative recommender. Basically, it will recommend a product based on how well it correlates with other products with respect to user actions. We apply it using SVD model to extract correlation from the item-to-item similarity matrix. These similarities are combined with the predictive model output to create recommendations. Usually, people browsing through the website and viewed a lot of products but they didn't purchase any product and the reason sometimes is the price. So, the recommender will check the predictive model output and if the prediction result was the customer will not purchase then it will recommend a similar product to what he viewed previously but with a lower price to encourage him to purchase the product.
#### Recommendation Functions Code
https://git.generalassemb.ly/DSI-MISK-VIII/DataArmors/blob/e4bd8cdb3b83701262834d0f0f637478d41dd934/inference.py#L48
https://git.generalassemb.ly/DSI-MISK-VIII/DataArmors/blob/e4bd8cdb3b83701262834d0f0f637478d41dd934/inference.py#L78
 ### Flask App
 Since models were built in Python scripts, a natural choice was to use [Flask]([http://flask.pocoo.org/](http://flask.pocoo.org/)) framework to implement our web application. The interactive application built on top of HTML and CSS. Models were exported to Pickle and H5 files.
 
 
 
 

# Result

Our recommender system composed of a predictive model and an item-based collaborative model. In the predictive, model we implemented a sequential deep learning algorithm; RNN. we use two types of RNN, simpleRNN and LSTM. We follow the many-to-one architecture in both algorithms. SimpleRNN gives us 96% accuracy and the loss function was 3.5% and LSTM gives us 95% accuracy and its loss function was 4.3%. we decide to continue with SimleRNN. We found out that simpleRNN is better than LSTM when dealing with a short sequence of data and has faster training, computationally less expensive than LSTM. So, we decide to continue using simpleRNN in model implementation. In the item-based collaborative model, SVD model used to calculate the similarity between products. The two models combined to build the recommender system. The following tables reprsent the accuracy scores for implemnted algorithms, and the overall scores of the predictive model: 
     
     
|   |loss|accuracy|validation loss|validation accuracy|  
|---|---|---|---|---|  
|SimpleRNN|0.0352|0.9651|0.0335|0.9668|  
|LSTM|0.0452|0.9581|0.0433|0.9611|
				  Table of all the scores of RNN algorithms
| |precision|recall|f1-score|support|  
|---|---|---|---|---|  
|accuracy|-|-|-|0.97|21842|  
|macro average|0.98|0.68|0.76|21842|  
|weighted average|0.97|0.97|0.96|21842|
				  Classification report of simpleRNN (implemnted model)


## Executive Summary:

We conducted a recommender system in the span of two weeks along with a predictive model, it recommended products based on user behavior. The predictive model used simpleRNN deep learning algorithm and LSTM algorithm but since LSTM gives us a lower score than simpleRNN. We decide to continue our implementation with simpleRNN, since it's better than LSTM when dealing with sequential data. We also, developed a web application using falsk framework and HTML. We used the following metrics to evaluate our predictive models: 1-loss. 2- accuracy. 3-validation loss. 4- validation accuracy. The scores as shown in the predictive model evaluation section above. The used data-set was from an online store called "Macy's". It contains information about cosmetics products such as the product's price, its brand, the customers like the user sessions and his ID, and the events such as the type of the event and the time when it happens.


## Future Works
 
- There are a few improvements that could be made to the recommender, such as :
    -   Covering the situation when the predictive model predict that a customer will purchase the product.
    -   Enhancing the execution time of the recommender system.
	-   Creating a hybrid recommendation system that blends both content and collaborative approaches.  
     -   Recommending Top five similar products.
    
    
    
## Resources 
 - Sequence-Aware Recommender Systems: [https://arxiv.org/pdf/1802.08452.pdf](https://arxiv.org/pdf/1802.08452.pdf)  
 - RNNs: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
 
 
## Acknowledgements
We would like to express our special thanks to our Instructors for their endless supports during this course:
 - Mukesh Mithrakumar
 - Husain Amer
 - Amal Alzamel 
 - Amjad Alsulami
 
 Also, we would like to express our thanks to General Assembly, Misk Academy and the Ministry of Communications and Information Technology of Saudi Arabia for thier generous sponsorsing our course.


