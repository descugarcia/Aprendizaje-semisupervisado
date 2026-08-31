Implementación de algunos algoritmos de aprendizaje semisupervisado.

- Co-training by Committee: A New Semi-supervised Learning Framework. https://doi.org/10.1109/ICDMW.2008.27
- Self-training semi-supervised classification based on density peaks of data. https://doi.org/10.1016/j.neucom.2017.05.072
- Learning with Local and Global Consistency. https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf
- Multilabel graph-based classification for missing labels. https://doi.org/10.1007/s00799-020-00295-3
- Tri-training: exploiting unlabeled data using three classifiers. https://doi.org/10.1109/TKDE.2005.186

# Ejemplo de uso

```
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier

from codigo.tritraining import TriTraining

X, y = sklearn.datasets.load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

model = DecisionTreeClassifier()
method_ins = TriTraining(model = model)
method_ins.fit(X_train, y_train, X_test)
pred = method_ins.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred), "Mcc:", matthews_corrcoef(y_test, pred))
```
