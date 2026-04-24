from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

model = GaussianNB()
model.fit(X, y)

predicted = model.predict(X)

print("Prediction:", predicted[:10])