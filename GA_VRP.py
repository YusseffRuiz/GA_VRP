import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt


class Sol: ### class holding the solution values
    def __init__(self):
        self.nodes_seq=None ### nodes ordered in the best sequence to send all packages
        self.obj=None
        self.fit=None ### best fitness function
        self.routes=None ### best routes selected
        self.numberVehicles=None ### number of vehicles used
        self.distTraveled = None  ### Distance Traveled in total
class Node: #### Node is used to get the data of each place in the excel sheet
    def __init__(self):
        self.id=0 ## number of place to visit (id = 0 is de depot or central station)
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0 ### how much the trucks can carry
class Model:
    def __init__(self):
        self.best_sol=None
        self.node_list=[]
        self.sol_list=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.opt_type=0
        self.vehicle_cap=0
        self.pc=0.5
        self.pm=0.2
        self.n_select=80
        self.popsize=100

class GA_VRP:
    def __init__(self):
        pass

    def readXlsxFile(self, filepath, model):
        #It is recommended that the vehicle depot data be placed in the first line of xlsx file
        node_seq_no =-1 #the depot node seq_no is -1,and demand node seq_no is 0,1,2,...

        df = pd.read_excel(filepath)

        ##The nodes will be filled according to the number of places located in the excel sheet
        for i in range(df.shape[0]):
            node=Node()
            node.id=node_seq_no
            node.seq_no=node_seq_no
            node.x_coord= df['x_coord'][i]
            node.y_coord= df['y_coord'][i]
            node.demand=df['demand'][i]
            if df['demand'][i] == 0:
                model.depot=node
            else:
                model.node_list.append(node)
                model.node_seq_no_list.append(node_seq_no)
            try:
                node.id=df['id'][i]
            except:
                pass
            node_seq_no=node_seq_no+1
        model.number_of_nodes=len(model.node_list)

        ##Initiating the model
    def genInitialSol(self, model):
        ## initialize first solution based on the first sequence of nodes from the model, same order as excel sheet
        for i in range(model.popsize):
            sol=Sol()
            sol.nodes_seq=copy.deepcopy(model.node_seq_no_list)
            model.sol_list.append(sol)

    def splitRoutes(self, nodes_seq, model):#### Important function to determine when each vehicle is full
        num_vehicle = 0
        vehicle_routes = []
        route = []
        remained_cap = model.vehicle_cap ## local variable to get each vehicle capacity
        for node_no in nodes_seq: ### which id of place is the vehicle going (will get to all places
            if remained_cap - model.node_list[node_no].demand >= 0: ### If the X vehicle still has capacity
                route.append(node_no) ## add the node to "completed places" list
                remained_cap = remained_cap - model.node_list[node_no].demand ## update how much capacity the vehicle has left
            else: ### when the vehicle is full, a new truck is needed
                vehicle_routes.append(route) ##The complete route is appended to the completed routes by all vehicles
                route = [node_no] ## start the route again with the last place id
                num_vehicle = num_vehicle + 1 ### next truck
                remained_cap = model.vehicle_cap - model.node_list[node_no].demand ## update the capacity on new truck
        vehicle_routes.append(route)### add last route to completed routes by all vehicles
        return num_vehicle,vehicle_routes



    def calDistance(self, route, model): ### where the actual distance is calculated
        ### used if we are looking to minimize distance traveled
        distance=0
        depot=model.depot

        ### The foor loop finds the total distance across all locations with the euclidean distance
        for i in range(len(route)-1):
            from_node=model.node_list[route[i]]
            to_node=model.node_list[route[i+1]]
            distance+=math.sqrt((from_node.x_coord-to_node.x_coord)**2+(from_node.y_coord-to_node.y_coord)**2)
        first_node=model.node_list[route[0]]
        last_node=model.node_list[route[-1]]
        ###Lastly the distance from the depot to the first place and the last place are added
        distance+=math.sqrt((depot.x_coord-first_node.x_coord)**2+(depot.y_coord-first_node.y_coord)**2)
        distance+=math.sqrt((depot.x_coord-last_node.x_coord)**2+(depot.y_coord - last_node.y_coord)**2)
        return distance

    def calFit(self, model):
        ####Fitness values, here is where we determine if this is the best solution yet or if this need to look for a better one
        ## We are trying to minimize either the number of vehicles or distance traveled
        #calculate fit value：fit=Objmax-obj
        Objmax=-float('inf') ### chosen as -infinite for the first value be the "furthest"
        best_sol=Sol()#record the local best solution, local variable to compare with the rest.
        best_sol.obj=float('inf')
        #计算目标函数
        for sol in model.sol_list:
            nodes_seq=sol.nodes_seq
            num_vehicle, vehicle_routes = self.splitRoutes(nodes_seq, model) ### get how many vehicles were needed to complete all routes
            distance = 0 ## initialize distance
            sol.routes = vehicle_routes  ###list of nodes in order for the routes covered
            sol.numberVehicles = num_vehicle

            for route in vehicle_routes:
                distance += self.calDistance(route, model)  ###where the actual distance is calculated
            sol.distTraveled = distance

            if model.opt_type==0: ### optimize based on less vehicles
                sol.obj=num_vehicle ### how many vehicles were used in this iteration
                if sol.obj>Objmax:
                    Objmax=sol.obj ### if the number of vehicles is higher than last time, previous value remains
                if sol.obj<best_sol.obj: ## if the number of vehicles is smaller, is chosen as best fitness function
                    best_sol=copy.deepcopy(sol)
            else: ## optimize based on less traveled distance
                sol.obj=distance
                if sol.obj>Objmax:
                    Objmax=sol.obj ### if the distance is higher than last time, previous value remains
                if sol.obj < best_sol.obj:## if the distance is smaller, is chosen as best fitness function
                    best_sol = copy.deepcopy(sol)
        #calculate fit value
        for sol in model.sol_list: ### model has different possible solutions, each is updated with the fitness function
            sol.fit=Objmax-sol.obj
        #update the global best solution
        if best_sol.obj<model.best_sol.obj: ### model best solution is updated if the solution found in this epoch was better than previous
            model.best_sol=best_sol
        #Binary tournament
    def selectSol(self, model):
        sol_list=copy.deepcopy(model.sol_list) ### local variable to store all solutions in the model
        model.sol_list=[] ### reload a new solution list to be filled by this function
        for i in range(model.n_select): ### based on how many excellent children we decided
            f1_index=random.randint(0,len(sol_list)-1)
            f2_index=random.randint(0,len(sol_list)-1)
            f1_fit=sol_list[f1_index].fit ### get a random fitness function from the solution list
            f2_fit=sol_list[f2_index].fit
            ##solution list is now filled with 10 random individuals selected based on their fitness value
            if f1_fit<f2_fit:
                model.sol_list.append(sol_list[f2_index])
            else:
                model.sol_list.append(sol_list[f1_index])
        #Order Crossover (OX)
        ### Next function works with binary values. From previous functions, the solution list is a list of 1 and 0s
        #this is because fitness model will be updated in the vehicle or distance different from one epoch to anothe
        #i.e. 1st iteration we are using 40 vehicles, if next iteration is found that best solution is 40, the fitness value
        # will be 0
        ## in contrast, if 2nd iteration is found that best solution is 39, fitness value is 1
    def crossSol(self, model): ### Here is where parents are combined to generate new children
        sol_list=copy.deepcopy(model.sol_list)
        model.sol_list=[]
        while True:
            f1_index = random.randint(0, len(sol_list) - 1) ## random index based on the size of solution list
            f2_index = random.randint(0, len(sol_list) - 1)
            if f1_index!=f2_index: # make sure both index are not the same
                ##children copied exactly the same as parent
                f1 = copy.deepcopy(sol_list[f1_index])
                f2 = copy.deepcopy(sol_list[f2_index])
                if random.random() <= model.pc:
                    cro1_index=int(random.randint(0,model.number_of_nodes-1))
                    cro2_index=int(random.randint(cro1_index,model.number_of_nodes-1))
                    ### first random number froom 0 to the size of the number of nodes
                    ### second random number from the first randomly generated number to the end
                    ##Each child is divided in three parts for crossover.
                    ###First and third part are generated randomly
                    new_c1_a = []
                    ##Second part is exactly the same as parents
                    new_c1_b=f1.nodes_seq[cro1_index:cro2_index+1]
                    new_c1_c = []
                    new_c2_a = []
                    new_c2_b=f2.nodes_seq[cro1_index:cro2_index+1]
                    new_c2_c = []
                    for index in range(model.number_of_nodes):

                        ##Generating first child
                        if len(new_c1_a)<cro1_index:
                            if f2.nodes_seq[index] not in new_c1_b: ### used to determine that the node is not being used again
                                new_c1_a.append(f2.nodes_seq[index]) ### if it hasn't been added, add the node to this new child
                        else:
                            if f2.nodes_seq[index] not in new_c1_b:
                                new_c1_c.append(f2.nodes_seq[index]) ### second part of first child in case the size is higher than first random generated number

                    ####same procedure but for second child
                    for index in range(model.number_of_nodes):
                        if len(new_c2_a)<cro1_index:
                            if f1.nodes_seq[index] not in new_c2_b:
                                new_c2_a.append(f1.nodes_seq[index])
                        else:
                            if f1.nodes_seq[index] not in new_c2_b:
                                new_c2_c.append(f1.nodes_seq[index])

                    ### This part is to combine 3 parts of the child generated in the previous lines
                    new_c1=copy.deepcopy(new_c1_a)
                    new_c1.extend(new_c1_b)
                    new_c1.extend(new_c1_c)
                    f1.nodes_seq=new_c1
                    new_c2=copy.deepcopy(new_c2_a)
                    new_c2.extend(new_c2_b)
                    new_c2.extend(new_c2_c)
                    f2.nodes_seq=new_c2
                    model.sol_list.append(copy.deepcopy(f1))
                    model.sol_list.append(copy.deepcopy(f2))
                else:
                    model.sol_list.append(copy.deepcopy(f1))
                    model.sol_list.append(copy.deepcopy(f2))
                if len(model.sol_list)>model.popsize:
                    break
        #mutation

        ##Used to change one bit in the chain of bits on the solution list generated by the crossover.
        ## The childs are mutated so diversity can be generated
    def muSol(self, model):
        sol_list=copy.deepcopy(model.sol_list)
        model.sol_list=[]
        while True:
            f1_index = int(random.randint(0, len(sol_list) - 1))
            f1 = copy.deepcopy(sol_list[f1_index])
            ###Generating which bit is going to be change randomly
            m1_index=random.randint(0,model.number_of_nodes-1)
            m2_index=random.randint(0,model.number_of_nodes-1)
            if m1_index!=m2_index:
                if random.random() <= model.pm: ### Determine if the mutation is going to occur with a probability of model.pm
                    node1=f1.nodes_seq[m1_index]
                    f1.nodes_seq[m1_index]=f1.nodes_seq[m2_index] ### mutate changing one bit of child 2 for 1 in child 1
                    f1.nodes_seq[m2_index]=node1
                    model.sol_list.append(copy.deepcopy(f1))
                else:
                    model.sol_list.append(copy.deepcopy(f1)) ### add the values to the solution list
                if len(model.sol_list)>model.popsize:
                    break
    def plotObj(self, obj_list): ### ploting the values we wanted to minimize, the objective function values
        plt.rcParams['axes.unicode_minus'] = False  # Show minus sign
        plt.plot(np.arange(1,len(obj_list)+1),obj_list)
        plt.xlabel('Iterations')
        plt.ylabel('Obj Value')
        plt.grid()
        plt.xlim(1,len(obj_list)+1)
        plt.show()
    def outPut(self, model, outFilepath): ####
        work=xlsxwriter.Workbook(outFilepath)
        worksheet=work.add_worksheet()
        worksheet.write(0,0,'opt_type')
        worksheet.write(1,0,'obj')
        if model.opt_type==0:
            worksheet.write(0,1,'number of vehicles')
        else:
            worksheet.write(0, 1, 'drive distance of vehicles')
        worksheet.write(1,1,model.best_sol.obj)
        for row,route in enumerate(model.best_sol.routes):
            worksheet.write(row+2,0,'v'+str(row+1))
            r=[str(i)for i in route]
            worksheet.write(row+2,1, '-'.join(r))
        work.close()


    def ga_main(self, filepath,epochs,pc,pm,popsize,n_select,v_cap,opt_type, outFile):

        """
                        :param filepath:Xlsx file path
                        :param epochs:Number of Iterations
                        :param pc:Crossover probability
                        :param pm:Mutation probability
                        :param popsize:Population size
                        :param n_select:Number of excellent individuals selected
                        :param v_cap:Vehicle capacity, how much every vechicle can carry
                        :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
                        :return:
                        """

        model = Model()
        model.vehicle_cap = v_cap
        model.opt_type = opt_type

        model.pc = pc
        model.pm = pm
        model.popsize = popsize
        model.n_select = n_select




        self.readXlsxFile(filepath, model)
        self.genInitialSol(model)
        history_best_obj = []
        best_sol=Sol()
        best_sol.obj=float('inf')
        model.best_sol=best_sol
        for ep in range(epochs):
            self.calFit(model)
            self.selectSol(model)
            self.crossSol(model)
            self.muSol(model)
            history_best_obj.append(model.best_sol.obj)
            print("%s/%s， best obj: %s" % (ep,epochs,model.best_sol.obj))
            print("Distance: %d, Vehicles: %s" % (model.best_sol.distTraveled, model.best_sol.numberVehicles))
        self.plotObj(history_best_obj)
        self.outPut(model, outFile)


