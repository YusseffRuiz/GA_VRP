from numpy.random import randint
from numpy.random import rand

##Tournament selection procedure, taking the population and returning one selected parent
##calling the function for each position create a list of parents

class GA_Implementation:

    def __int__(self):
        pass

    def selection(self, pop, scores, k = 3):
        #random selection
        selection_i = randint(len(pop))
        for i in randint(0,len(pop), k-1):
                #tournament selection
            if scores[i] < scores[selection_i]:
                selection_i = i
        return pop[selection_i]

    ##Crossover function
    #From two parents, two children are created based on their data

    def crossover(self, p1, p2, r_cross):
        ###initiate the children to be same size as parents
        c1, c2 = p1.copy(), p2.copy()
        ##this random is to determine if the crossover must be done.
        ## otherwise, children are exact copy of parents
        if rand() < r_cross:
            pt = randint(1, len(p1) - 2)
            ### childrens are a crossover of the 2 parents determined by a random number
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]

    ##mutation is a variation on the children so they differ from their parents to create diversity
    ##r_mut is a hyperparamenter probability value
    def mutation(self, bitString, r_mut):
        for i in range(len(bitString)):
            ##check for a mutation
            if rand() < r_mut:
                bitString[i] = 1 - bitString[i]

    def decode(self, bounds, n_bits, bitString):
        decoded = list()
        largest = 2**n_bits
        for i in range(len(bounds)):
            ##extract the substring
            start, end = i * n_bits, (i*n_bits)+n_bits
            substring = bitString[start:end]
            ##convert bitstring to a string of chars
            chars = ''.join([str(s) for s in substring])
            ##convert string to integer
            integer = int(chars, 2)
            ##scale integer to desired range
            value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
            ##store
            decoded.append(value)
        return decoded

    def ga_main(self, objective, n_bits, n_iter, n_pop, r_cross, r_mut, select, bounds = [[-1.0, 1.0], [-1.0, 1.0]]):
        if(select.lower() == "cont"):
            ##continuous function
            pop = [randint(0,2,n_bits*len(bounds)).tolist() for _ in range(n_pop)]
            best, best_eval = 0, objective(self.decode(bounds, n_bits, pop[0]))
        else: ##binary
            pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
            best, best_eval = 0, objective(pop[0])



        for gen in range(n_iter):
            ##evaluate all candidates in population

            if (select.lower() == "cont"):
                decoded = [self.decode(bounds, n_bits, p) for p in pop]
                scores = [objective(d) for d in decoded]
            else: ##"binary"
                scores = [objective(c) for c in pop]

            for i in range(n_pop):
                ##Fitness evaluation, getting the best value
                ##if we require to minimize, change the sign to <
                ##if we require to maximize, change the sign to >
                if scores[i] < best_eval:
                    if (select.lower() == "cont"):
                        best, best_eval = decoded[i], scores[i]
                        print("%d, new best f(%s) = %.3f" % (gen, decoded[i], scores[i]))
                    else:
                        best, best_eval = pop[i], scores[i]
                        print("%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))

            ##select parents
            selected = [self.selection(pop, scores) for _ in range(n_pop)]

            children = list()
            for i in range(0, n_pop, 2):
                ##get selected parents in pairs
                p1, p2 = selected[i], selected[i + 1]
                ##crossover and mutation
                for c in self.crossover(p1, p2, r_cross):
                    self.mutation(c, r_mut)
                    ##store for next generation
                    children.append(c)
            pop = children
        return best, best_eval

