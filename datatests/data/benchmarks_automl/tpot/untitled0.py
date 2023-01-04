from tpot import TPOT
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOT(generations=5,verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_iris_pipeline.py')
