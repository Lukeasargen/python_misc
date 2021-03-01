import matplotlib.pyplot as plt
import numpy as np


def get_purchase_info(size, cost, budget):
    """ size = number of prop
        cost = cost per prop
        budget = total money
    """

    tot_h = int(budget / cost)  # total houses
    base_h = tot_h // size  # houses on each prop
    extra_p = tot_h % size  # props with extra houses
    base_p = size - extra_p  # props with base houses
    extra_h = 0 if ((tot_h - size*base_h) == 0) else base_h + 1
    # print("tot_h :", tot_h)
    # print("base_p :", base_p)
    # print("base_h :", base_h)
    # print("extra_p :", extra_p)
    # print("extra_h :", extra_h)

    out_str = ""

    if tot_h == 0:
        out_str = "You cannot afford even one house."
    elif (tot_h)>=(5*size):
        out_str = "{} will have a hotel".format(size)
    elif extra_p == 0:
        out_str = "{} will have {}".format(base_p, base_h)
    elif base_h == 0:
        out_str = "{} will have {}".format(extra_p, extra_h)
    elif extra_h == 5:
        out_str = "{} will have {} and 1 will have a hotel".format(base_p, base_h)
    else:
        out_str = "{} will have {} and {} will have {}".format(base_p, base_h, extra_p, extra_h)

    return [base_p, base_h, extra_p, extra_h], out_str

if __name__ == "__main__":
    size = 3
    cost = 50

    for i in range(0, 2000, 50):
        data, output = get_purchase_info(size, cost, i)
        print(i, data, output)


    exit()
    
    budget = list(range(0, 2000, 10))
    data = np.array([get_purchase_info(size, cost, b)[0] for b in budget])
    line_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    line_labels = [
        "base props",
        "base houses",
        "extra props",
        "extra houses"
    ]
    for i in range(len(line_labels)):
        plt.plot(budget, data[:, i], color=line_colors[i], label=line_labels[i])
    plt.legend()
    plt.show()
    exit()
