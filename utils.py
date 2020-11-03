from sklearn.metrics import f1_score, accuracy_score


class EvaluationMetric:
    def __init__(self, opt):
        self.n_classes = opt.n_classes
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def get_accuracy(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc

    def get_f1_score(self):
        labels = list(range(self.n_classes))
        f1score = f1_score(self.y_true, self.y_pred,
                           average='weighted', labels=labels)
        return f1score
