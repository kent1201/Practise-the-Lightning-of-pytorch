def compute_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 20: # 10
        return 0.5

    elif epoch == 60: # 50
        return 0.25

    elif 80 < epoch < 109: # 75, 99
        return 0.01