
def func_ret_tuple():
    return 1, 2

def total():
    totals = 0, 0

    for i in range(100):

        totals = map(lambda x, y: x + y, totals, func_ret_tuple())

    return totals

value1, value2 = total()

print('totals[0]', value1)
print('totals[1]', value2)