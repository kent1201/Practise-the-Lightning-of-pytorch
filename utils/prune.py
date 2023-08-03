def compute_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 10:
        return 0.5

    elif epoch == 50: # 50
        return 0.25

    elif 75 < epoch < 99: # 75, 99
        return 0.01