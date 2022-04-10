from sklearn.datasets import load_wine
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Load Wine UCI Data from scikit-learn
wine = load_wine()

# Assign training and testing data
x = wine.data
y = wine.target

# Training test split
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Decision tree pruned at 4 layers
decision_tree = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
# Train decision tree
decision_tree = decision_tree.fit(x_train, y_train)
# Visualize decision tree
tree.plot_tree(decision_tree.fit(x_train, y_train))

# Predicted values from test data
y_pred = decision_tree.predict(x_test)

# Classification problem use confusion matrix for accuracy
c = confusion_matrix(y_test, y_pred)
diagonal = np.diagonal(c)
accuracy = np.sum(diagonal) / np.sum(c)
print('Confusion Matrix \n', c)
print('Testing Accuracy: {:%}'.format(accuracy))
