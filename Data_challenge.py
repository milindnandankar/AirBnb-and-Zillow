"""
@author: Milind Nandankar
contact: milindnandankar@gmail.com
"""

import pandas as pd
import numpy as np
from uszipcode import ZipcodeSearchEngine
from re import sub
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#get city for which analysis to be performed
city = input('Enter City:')


#set occupancy to 75% as given in proglem statement
occupancy_rate = .75


#read airbnb data
airbnb_df = pd.read_csv('listings.csv',engine='python')


#drop unwanted columns
airbnb_df = airbnb_df.iloc[:,39:70]


#setting static column names so that if new input file is used with different column names, code should still work.
airbnb_df.columns = ['neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city',
       'state', 'zipcode', 'market', 'smart_location', 'country_code',
       'country', 'latitude', 'longitude', 'is_location_exact',
       'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms',
       'beds', 'bed_type', 'amenities', 'square_feet', 'price', 'weekly_price',
       'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'maximum_nights', 'calendar_updated']

	   
#rows with only two bedrooms
airbnb_df = airbnb_df[airbnb_df['bedrooms']==2]


#remove rows with missing zipode
airbnb_df = airbnb_df[airbnb_df.notnull()['zipcode']]


#convert prices to float format
airbnb_df['price'] = airbnb_df['price'].apply(lambda x:float(sub(r'[^\d.]', '', x)))


#create function to get list of zipcode for New York
search = ZipcodeSearchEngine()
def get_zipcode(City):
    #remove white spaces
    City = City.lower().replace(' ','')
    res = search.by_city(city=City, returns=0)
    res = pd.DataFrame(res)[0].to_frame()
    res.columns = ['zipcode']
    return(res)


#get list of zipcodes for city
zip = get_zipcode(city)


#filter data for given city
airbnb_city_df = airbnb_df.merge(zip, on='zipcode', how='inner')


#get median prices for each zipcode
per_day_prices = airbnb_city_df.groupby('zipcode').median().reset_index()[['zipcode','price']]


#count the number of rows for each zipcode
count = airbnb_city_df.groupby('zipcode').count().reset_index()[['zipcode','price']]


#consider only those zipcode for which number of rows are greater than or equal to 10. Zipcodes with number of rows below 10 don't have enough data for analysis.
zip_to_keep = pd.DataFrame(count[count['price']>=10]['zipcode'])


#filter prices for remaining zipcodes.
per_day_prices = per_day_prices.merge(zip_to_keep, on='zipcode', how='inner')


#filter data for remaining zipcodes.
airbnb_city_df = airbnb_city_df.merge(zip_to_keep, on='zipcode', how='inner')


#read data from Zillow
zillow_df = pd.read_csv('Zip_Zhvi_2bedroom.csv')


#convert zipcodes to string format.
zillow_df['RegionName'] = zillow_df['RegionName'].apply(str)


#merge Zillow data with Airbnb data
zillow_city_df = zillow_df.merge(per_day_prices, left_on=['RegionName'],right_on=['zipcode'], how='inner')


#filter zipcodes for which Zillow as well as Airbnb data is available
final_zipcodes = pd.DataFrame(zillow_city_df['RegionName'].unique())


#change RegionName column to zipcode for ease of use
final_zipcodes.columns = ['zipcode']


#sort data by price in descending order
zillow_city_df = zillow_city_df.sort_values(by='price', ascending=False).reset_index(drop=True)


#plot Airbnb median prices for zipcodes
sns.set(style="whitegrid")
fig, ax = plt.subplots()
fig.set_size_inches(9, 4.5)
ax = sns.barplot(x='zipcode' , y='price' , data = zillow_city_df, palette="Blues_d", order = zillow_city_df['zipcode'].tolist())
ax.set_title('Per Day Median Prices')
plt.xlabel("Zipcode")
plt.ylabel("Price")
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set_xticklabels(zillow_city_df.zipcode)
#ax.patch.set_facecolor('#FFFFFF')
#ax.spines['bottom'].set_color('#CCCCCC')
#ax.spines['bottom'].set_linewidth(1)
#ax.spines['left'].set_color('#CCCCCC')
#ax.spines['left'].set_linewidth(1)
filename = 'img1' + '.png' 
plt.savefig(filename,dpi=450)


#filter airbnb data as well
airbnb_city_df = airbnb_city_df.merge(final_zipcodes, on='zipcode')


#plot box plot for prices
fig, ax = plt.subplots()
fig.set_size_inches(9, 4.5)
sns.boxplot(x='zipcode', y='price', data=airbnb_city_df, showfliers= False,palette="Blues_d", order = zillow_city_df['zipcode'].tolist())
ax = plt.gca()
ax.set_title('Price Variations' )
plt.xlabel("Zipcode")
plt.ylabel("Price")
filename = 'img2' + '.png' 
plt.savefig(filename,dpi=450)


#unpivote data for last 60 months for furthur calculation.
prices_new = pd.melt(zillow_city_df.iloc[:,-62:-1], id_vars=['zipcode'], value_vars=zillow_city_df.iloc[:,-62:-1].columns.tolist()[:-1], var_name='date', value_name='cost')


#extract year from date
prices_new['year'] = prices_new['date'].apply(lambda x: x[:4])


#get mean of prices
year_prices_new = prices_new.groupby(by=['zipcode', 'year']).mean()


#reset index
year_prices_new = year_prices_new.reset_index()


#plot trend in cost of houses in last 60 months
sns.lmplot(x="year", y="cost", data=year_prices_new.astype(int), hue='zipcode', palette="Blues_d",size=6,aspect=1.5,order=2)
ax = plt.gca()
ax.title.set_position([.5, .995])
ax.set_title('Cost of Two Bedroom House Trend' )
plt.xlabel("Year of Purchase")
plt.ylabel("Cost of a House")
filename = 'img3' + '.png' 
plt.savefig(filename,dpi=450)


#calculate return on investment
profit = year_prices_new.merge(zillow_city_df[['zipcode','price']],on='zipcode')
profit['roi_yearly'] = profit['price']/profit['cost']*365*occupancy_rate*1000


#plot return on investment trend for last 60 months
sns.lmplot(x="year", y="roi_yearly", data=profit.astype(int), hue='zipcode', palette="Blues_d",size=6,aspect=1.5,order=2)
ax = plt.gca()
ax.title.set_position([.5, .995])
ax.set_title('Return on Investment Trend' )
plt.tick_params(labelleft='off')
plt.xlabel("Year of Purchase")
plt.ylabel("% Return on Investment Y-Y")
filename = 'img4' + '.png' 
plt.savefig(filename,dpi=450)


#predict return on invetment for current year
reg2 = LinearRegression()
current_year = datetime.datetime.now().year
prediction = pd.DataFrame([])
for i in range (0,final_zipcodes.shape[0]):
    zipcode_i = final_zipcodes['zipcode'][i]
    data = profit[profit['zipcode']==zipcode_i]
    poly_reg = PolynomialFeatures(degree = 2)
    X_poly = poly_reg.fit_transform(data['year'].reshape(-1, 1))
    y = data['roi_yearly']
    reg2.fit(X_poly, y)
    prediction = prediction.append(pd.DataFrame([zipcode_i, float(reg2.predict(poly_reg.transform(current_year)))]).transpose())

prediction.columns = ['zipcode','pred']
prediction['pred'] = prediction['pred']/10
prediction = prediction.sort_values(by='pred',ascending=False).reset_index(drop=True)


#plot return on investment predictions for current year
fig, ax = plt.subplots()
fig.set_size_inches(9, 4.5)
ax = sns.barplot(x='zipcode' , y='pred' , data = prediction, palette="Blues_d", order = prediction['zipcode'].tolist())
ax.set_title('Return on Investment if Bought in '+ str(current_year))
plt.xlabel("Zipcode")
plt.ylabel("% Return on Investment Y-Y")
ax.set_xticklabels(prediction.zipcode)
filename = 'img5' + '.png' 
plt.savefig(filename,dpi=450)


#creting HTML page to display all viz

filename = 'Data_challenge.html'
f = open(filename,'w')

message = """
<h1 style="text-align: center;"><span style="text-decoration: underline;">Airbnb-Zillow Data Challenge</span></h1>
<h2>AirBnb: Per Day Median Prices</h2>
<div><img style="width: 900px; height: 450px;" src="img1.png" /></div>
<div>&nbsp;</div>
<p>&nbsp;</p>
<h2>AirBnb: Per Day Price Variation</h2>
<div><img style="width: 900px; height: 450px;" src="img2.png" /></div>
<div>&nbsp;</div>
<div>
<h2>Zillow: Cost of Two Bedroom Houses</h2>
<div><img style="width: 900px; height: 600px;" src="img3.png" /></div>
<div>&nbsp;</div>
<div>
<h2>Return on Investment</h2>
<div><img style="width: 900px; height: 600px;" src="img4.png" /></div>
<div>&nbsp;</div>
<div>
<h2>Return on Investmet for Current Year </h2>
<div><img style="width: 900px; height: 450px;" src="img5.png" /></div>
</div>
</div>
</div>"""


f.write(message)
f.close()

webbrowser.open_new_tab(filename)

