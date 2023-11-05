import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

# plot results
def _plot_results(y_test, y_pred):
    class_names = ['0', '1']
    _, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap= "YlGnBu" , fmt = 'g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y = 1.1)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    print(metrics.confusion_matrix(y_test, y_pred))
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))

def Plotter(y_test):
    return lambda y_pred : _plot_results(y_test, y_pred)
