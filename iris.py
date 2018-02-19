from sklearn import neighbors
from sklearn import datasets

iris = datasets.load_iris()

clf = neighbors.KNeighborsClassifier()

clf.fit(iris.data, iris.target)

sepal_l = float(input("Enter the sepal length of the flower: "))
sepal_w = float(input("Enter the sepal width of the flower: "))
petal_l = float(input("Enter the petal length of the flower: "))
petal_w = float(input("Enter the petal width of the flower: "))

if clf.predict([[sepal_l, sepal_w, petal_l, petal_w]]) == 0:
	print("Setosa")
elif clf.predict([[sepal_l, sepal_w, petal_l, petal_w]]) == 1:
	print("Versicolor")
elif clf.predict([[sepal_l, sepal_w, petal_l, petal_w]]) == 2:
	print("Virginica")