#https://deepblade.com/artificial-intelligence/machine-learning/what-is-confusion-matrix-accuracy-precision-recall-f1-score/

'''
in a Classification model confusion matrix provide detailed
description of the performance of the model.
'''


actual = [1,1,1,0,0,1,0,0,0,1,0,1,1,0,0]
predicted = [0,1,1,0,0,1,1,0,1,1,0,1,1,0,1]

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(actual, predicted)
print(conf_matrix)


print ("Using pandas")

#Compute classification report with Accuracy, Precision, Recall, F1 score

actual = [1,1,1,0,0,1,0,0,0,1,0,1,1,0,0]
predicted = [0,1,1,0,0,1,1,0,1,1,0,1,1,0,1]

import pandas as pd
from sklearn.metrics import classification_report
report = pd.DataFrame(classification_report(actual, predicted, output_dict=True))
print(report)

