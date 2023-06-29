from lion_pytorch import Lion

def LionOptimizer(params, lr=1e-4, weight_decay=1e-2):
    return Lion(params, lr=lr, weight_decay=weight_decay)