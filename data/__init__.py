from data.Classification.classification_dataset import ClassificationDataset


def CreateDataset(args):
    task = args.task
    if task == "classification":
        return ClassificationDataset(args=args)
    else:
        raise ValueError("No such task {}".format(task))