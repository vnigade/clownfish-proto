from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class EvaluationMetric:
    def __init__(self, opt):
        self.n_classes = opt.n_classes
        self.y_pred = []
        self.y_true = []
        self.labels = list(range(opt.n_classes))

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def get_accuracy(self):
        acc = accuracy_score(self.y_true, self.y_pred)
        return acc

    def get_f1_score(self):
        f1score = f1_score(self.y_true, self.y_pred,
                           average='weighted', labels=self.labels)
        return f1score

    def get_precision_score(self):
        precision = precision_score(self.y_true, self.y_pred)
        return precision

    def get_recall_score(self):
        recall = recall_score(self.y_true, self.y_pred)
        return recall
