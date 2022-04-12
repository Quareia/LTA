import torch
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


@torch.no_grad()
def accuracy_fn(y_true, y_pred):
    """Micro P = Micro R = Micro F1 = Accuracy."""
    assert len(y_pred) == len(y_true)

    return f1_score(y_true, y_pred, average='micro', zero_division=0)


@torch.no_grad()
def precision_fn(y_true, y_pred):
    """Precision Score."""
    assert len(y_pred) == len(y_true)

    return precision_score(y_true, y_pred, average='macro', zero_division=0)


@torch.no_grad()
def recall_fn(y_true, y_pred, average='macro'):
    """Recall Score."""
    assert len(y_pred) == len(y_true)

    return recall_score(y_true, y_pred, average=average, zero_division=0)


@torch.no_grad()
def macro_f1_fn(y_true, y_pred):
    """F1 score."""
    assert len(y_pred) == len(y_true)

    return f1_score(y_true, y_pred, average='macro', zero_division=0)


@torch.no_grad()
def HM_fn(x, y, eps=1e-6):
    """Harmonic Mean."""
    return 2 * (x * y) / (x + y + eps)


@torch.no_grad()
def top_k_acc_fn(y_true, output, k=3):
    assert len(output) == len(y_true)
    correct = 0
    for i in range(k):
        correct += torch.sum(output[:, i] == y_true).item()
    return correct / len(y_true)


@torch.no_grad()
def seen_metric(y_true, y_pred, num_seen_class):
    res = {}
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    p = 0.0
    r = 0.0
    f1 = 0.0
    c = 0
    for i in range(num_seen_class):
        if str(i) not in report:
            c += 1
            continue
        p += report[str(i)]['precision']
        r += report[str(i)]['recall']
        f1 += report[str(i)]['f1-score']

    res['seen_accuracy'] = accuracy_fn(y_true, y_pred)
    res['seen_precision'] = p / (num_seen_class - c)
    res['seen_recall'] = r / (num_seen_class - c)
    res['seen_macro_f1'] = f1 / (num_seen_class - c)
    return res


@torch.no_grad()
def unseen_metric(y_true, y_pred, num_seen_class):
    res = {}
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    p = 0.0
    r = 0.0
    f1 = 0.0
    c = 0
    for i in range(num_seen_class):
        if str(i) not in report:
            c += 1
            continue
        p += report[str(i)]['precision']
        r += report[str(i)]['recall']
        f1 += report[str(i)]['f1-score']

    res['unseen_accuracy'] = accuracy_fn(y_true, y_pred)
    res['unseen_precision'] = p / (num_seen_class - c)
    res['unseen_recall'] = r / (num_seen_class - c)
    res['unseen_macro_f1'] = f1 / (num_seen_class - c)
    return res


@torch.no_grad()
def calc_result_from_report(eval_type, report, n_class, class_start, y_t, y_p):
    """Calculate result dict from sklearn classification report

    Args:
        eval_type: (str) result type (only seen/ZSL/GZSL)
        report: (dict)  classification_report() results
        n_class: (int) the number of classes for evaluation
        class_start: (int) the start of class index
        y_t: (Tensor) y true
        y_p: (Tensor) y pred

    Returns:
        res: Results in a Dict.
    """
    res = {}
    p = 0.0
    r = 0.0
    f1 = 0.0

    not_mention_class = 0  # Some classes may be not included in test set.
    for i in range(class_start, class_start + n_class):
        idx = str(i)
        if idx not in report:
            not_mention_class += 1
            continue
        p += report[idx]['precision']
        r += report[idx]['recall']
        f1 += report[idx]['f1-score']
    # print(c)

    res['{}_accuracy'.format(eval_type)] = accuracy_fn(y_t, y_p)
    res['{}_precision'.format(eval_type)] = p / (n_class - not_mention_class)
    res['{}_recall'.format(eval_type)] = r / (n_class - not_mention_class)
    res['{}_macro_f1'.format(eval_type)] = f1 / (n_class - not_mention_class)
    return res


@torch.no_grad()
def all_metric(y_true, logit_pred, n_seen_test, n_seen_class, n_unseen_class):
    """The main metric calculation function.

    Args:
        y_true: (Tensor) true label
        logit_pred: (Tensor) predict logits
        n_seen_test: (int) the number of seen test set for evaluation
        n_seen_class: (int) the number of seen class for evaluation
        n_unseen_class: (int) the number of unseen class for evaluation

    Returns:
        Results in a Dict.
    """
    res = {}
    # only seen
    y_t = y_true[:n_seen_test]
    y_p = logit_pred[:n_seen_test, :n_seen_class].max(dim=1)[1]
    report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
    res.update(calc_result_from_report('only_seen', report, n_seen_class, 0, y_t, y_p))

    # only ZSL
    y_t = y_true[n_seen_test:] - n_seen_class
    y_p = logit_pred[n_seen_test:, n_seen_class:].max(dim=1)[1]
    report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
    res.update(calc_result_from_report('only_unseen', report, n_unseen_class, 0, y_t, y_p))

    # GZSL
    y_t = y_true
    y_p = logit_pred.max(dim=1)[1]
    report = classification_report(y_t, y_p, output_dict=True, zero_division=0)
    # # seen
    seen_y_t = y_t[:n_seen_test]
    seen_y_p = y_p[:n_seen_test]
    seen_recall = recall_fn(torch.ones_like(seen_y_t),
                                         seen_y_p < n_seen_class,
                                         average='binary')
    res.update(calc_result_from_report('seen', report, n_seen_class, 0, seen_y_t, seen_y_p))

    # # unseen
    unseen_y_t = y_t[n_seen_test:]
    unseen_y_p = y_p[n_seen_test:]

    unseen_recall = recall_fn(torch.ones_like(unseen_y_t),
                              unseen_y_p >= n_seen_class,
                              average='binary')
    res.update(calc_result_from_report('unseen', report, n_unseen_class, n_seen_class, unseen_y_t, unseen_y_p))

    # # GZSL
    res['GZSL_seen_recall'] = seen_recall  # domain binary classification
    res['GZSL_unseen_recall'] = unseen_recall  # domain binary classification
    res['GZSL_acc_hm'] = HM_fn(res['seen_accuracy'], res['unseen_accuracy'])
    res['GZSL_f1_hm'] = HM_fn(res['seen_macro_f1'], res['unseen_macro_f1'])
    return res, report


if __name__ == '__main__':
    # y_true = torch.tensor([1,2,3,4,5])
    # y_pred = torch.tensor([4,1,2,4,1])
    #
    # print(top_k_acc_fn(y_true, y_pred))
    pass
