def compute_amount(epoch):
    # the sum of all returned values need to be smaller than 1
    if epoch == 15:
        return 0.5

    elif epoch == 40: # 50
        return 0.25

    elif 75 < epoch < 99: # 75, 99
        return 0.125 # 0.01