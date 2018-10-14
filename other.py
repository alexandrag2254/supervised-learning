"""
	data is amount spent on advertising for each feature(tv, radio, newspaper) in thousands of dollars
	sales is the target/response - the amount of sales made in thousands of items
	response is continuous it is a regression supervised learning problem
"""

data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col = 0)
data.head()
data.tail()

"""
	Other datasets to explore
"""
# from sklearn.datasets import load_breast_cancer -> preloaded dataset in conda
# data = load_breast_cancer()
# This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets. https://goo.gl/U2Uwz2


# data2 = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Heart.csv', index_col = 0)
# Auto.csv
# College.csv
# Credit.csv

"""
	Shows 3 inline scatter plot of amount of money spent on feature vs the amount of sales 
	This indicates the relationship between the variables for each feature
	* kind="reg" is included to fit a line to plot 
	In this example, appears that TV shows the most correlation while newspaper shows the least correlation
"""
%matplotlib inline
sns.pairplot(data, x_vars=["TV", "radio", "newspaper"], y_vars="sales", size=7, aspect = 0.7, kind="reg")

# preparing X and y 
feature_cols = ["TV", "radio", "newspaper"]
X = data[feature_cols]
X.head()

y = data["sales"]
y.head()
# y.shape() should be single column 

"""
	Split X and Y into Training and Testing data sets 
"""
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.3, random_state = 1)

print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

"""
	example response:
		(150, 3)
		(150,)
		(50, 3)
		(50,)
"""
