from train.classify_trainer import Classifier

def CreateModel(args):
    if args.task == "classification":
        model = Classifier(args=args)
        return model
    else:
        raise ValueError("No such task {}".format(args.task))