# This is le Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import GA_Implementation
import GA_VRP


def onemax(x):
    return -sum(x)

def objective(x):
    return x[0]**2.0 + x[1]**2.0

def funcOptim(x):
    return (4*x[0] - 2*x[1]  + 7*x[2] + 5*x[3] + 11*x[4] + 1*x[5])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    ##bounds are based on the restrains that the function must have, otherwise, it will run indifinetely
    bounds1 = [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]
    n_iter = 300
    n_bits = 16
    n_pop = 100
    r_cross = 0.9
    r_mut = 1.0/(float(n_bits) * len(bounds))
    ga_algorithm = GA_Implementation.GA_Implementation()
    ###select = "binary" or "cont"
    best, score = ga_algorithm.ga_main(funcOptim, n_bits, n_iter, n_pop, r_cross, r_mut, "cont", bounds1)
    print("Done!")
    print('f(%s) = %f' % (best, score))
    """
    file='./Files/cvrp.xlsx'
    outFile = './Files/result.xlsx'
    ga_algorithm = GA_VRP.GA_VRP()

    ga_algorithm.ga_main(file,350,0.8,0.2,100,50,80,1, outFile)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
