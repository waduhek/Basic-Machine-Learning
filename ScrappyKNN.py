from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a, b)

class ScrappyKNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0

		for i in range(1, len(self.X_train)):
			dist = euc(row, X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i

		return self.y_train[best_index]

	def predict(self, X_test):
		predictions = []

		for row in X_test:
			label = self.closest(row)
			predictions.append(label)

		return predictions

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

clf = ScrappyKNN()

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

print(accuracy_score(y_test, predictions))