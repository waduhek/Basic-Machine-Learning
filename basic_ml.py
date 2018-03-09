from sklearn import tree

# 1 for bumpy texture, 0 for smooth texture
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 1 for orange, 0 for apple
labels = [1, 1, 0, 0]

clf = tree.DecisionTreeClassifier()

clf = clf.fit(features, labels)

print(clf.predict([[160, 1]]))
