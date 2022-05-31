import pandas as pd
import math
import random
import numpy as np
import copy
import xlsxwriter
import matplotlib.pyplot as plt
class Sol():
    def __init__(self):
        def __init__(self):
            self.nodes_seq = None  ### nodes ordered in the best sequence to send all packages
            self.obj = None
            self.routes = None  ### best routes selected
            self.numberVehicles = None  ### number of vehicles used
            self.distTraveled = None  ### Distance Traveled in total
            # self.cost= None
class Node():
    def __init__(self):
        self.id=0
        self.seq_no=0
        self.x_coord=0
        self.y_coord=0
        self.demand=0
class Model():
    def __init__(self):
        self.best_sol=None
        self.node_list=[]
        self.sol_list=[]
        self.node_seq_no_list=[]
        self.depot=None
        self.number_of_nodes=0
        self.opt_type=0
        self.vehicle_cap=0
        self.distance={}
        self.popsize=100
        self.alpha=2
        self.beta=3
        self.Q=100
        self.rho=0.5
        self.tau={}


class ACO_VRP:
    def __init__(self):
        pass
    def readXlsxFile(self, filepath,model):
        node_seq_no = -1
        df = pd.read_excel(filepath)
        for i in range(df.shape[0]):
            node=Node()
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
                node.name=df['name'][i]
            except:
                pass
            try:
                node.id=df['id'][i]
            except:
                pass
            node_seq_no=node_seq_no+1
        model.number_of_nodes=len(model.node_list)

    def initParam(self, model):
        for i in range(model.number_of_nodes): ### gest the distance between all the nodes and add tau variable in the model
            for j in range(i+1,model.number_of_nodes):
                d=math.sqrt((model.node_list[i].x_coord-model.node_list[j].x_coord)**2+
                            (model.node_list[i].y_coord-model.node_list[j].y_coord)**2)
                model.distance[i,j]=d
                model.distance[j,i]=d
                model.tau[i,j]=model.Q
                model.tau[j,i]=model.Q
    def movePosition(self, model): ## Primary function.
        ## Used as the method of moving the different ants to all possible positions to find the optimal path.
        sol_list=[]
        local_sol=Sol()
        local_sol.obj=float('inf')
        for k in range(model.popsize):
            #Random ant position
            ## With each generation, a set of ants are randomly placed on the available nodes. Then each ant create a solution by choosing next location based on probaility.

            nodes_seq=[int(random.randint(0,model.number_of_nodes-1))]
            all_nodes_seq=copy.deepcopy(model.node_seq_no_list)
            all_nodes_seq.remove(nodes_seq[-1])
            #Determine the next moving position according to pheromone
            while len(all_nodes_seq)>0:
                ##The next position to start building a solution is decided on the pheromone amount and a probability function.
                next_node_no=self.searchNextNode(model,nodes_seq[-1],all_nodes_seq)
                nodes_seq.append(next_node_no)
                all_nodes_seq.remove(next_node_no) ### performed until all nodes have been exhausted
            sol=Sol()
            sol.nodes_seq=nodes_seq
            sol.obj,sol.routes, sol.numVehicle, sol.distanceTraveled=self.calObj(nodes_seq,model) ### getting the best value, 0 for number of vehicles, 1 for distance
            sol_list.append(sol)
            ##storing local variable solution for each ant
            if sol.obj<local_sol.obj:
                local_sol=copy.deepcopy(sol)
        ##only the best solution from all ants
        model.sol_list=copy.deepcopy(sol_list)
        if local_sol.obj<model.best_sol.obj:
            model.best_sol=copy.deepcopy(local_sol)
    def searchNextNode(self, model,current_node_no,SE_List):
        prob=np.zeros(len(SE_List))
        ### Get the probability value to choose next node
        for i,node_no in enumerate(SE_List):
            eta=1/model.distance[current_node_no,node_no] ### inverse of the travel cost, in this case, the inverse of the distance
            ## (1/distance of the node)
            tau=model.tau[current_node_no,node_no] ### pheromone update
            prob[i]=((eta**model.alpha)*(tau**model.beta)) ### equation of probability (eta^alpha) * (tau^beta)
        #use Roulette to determine the next node
        cumsumprob=(prob/sum(prob)).cumsum() ### Cumsum is the cumulative sum of all values in the probability variable
        cumsumprob -= np.random.rand() ###
        next_node_no= SE_List[list(cumsumprob > 0).index(True)]
        return next_node_no ### Roulette determines a random probability value calculated on the function
    def upateTau(self, model):
        ##updating TAU according to a decay value, each generation, the pheromone level lowers, so it needs to be updated according to this
        rho=model.rho
        for k in model.tau.keys():
            model.tau[k]=(1-rho)*model.tau[k] ## decay*tau
        #update tau according to sol.nodes_seq(solution of TSP)
        for sol in model.sol_list:
            nodes_seq=sol.nodes_seq
            for i in range(len(nodes_seq)-1):
                from_node_no=nodes_seq[i]
                to_node_no=nodes_seq[i+1]
                model.tau[from_node_no,to_node_no]+=model.Q/sol.obj

    def splitRoutes(self, nodes_seq, model):  #### Important function to determine when each vehicle is full
        num_vehicle = 0
        vehicle_routes = []
        route = []
        remained_cap = model.vehicle_cap  ## local variable to get each vehicle capacity
        for node_no in nodes_seq:  ### which id of place is the vehicle going (will get to all places
            if remained_cap - model.node_list[node_no].demand >= 0:  ### If the X vehicle still has capacity
                route.append(node_no)  ## add the node to "completed places" list
                remained_cap = remained_cap - model.node_list[
                    node_no].demand  ## update how much capacity the vehicle has left
            else:  ### when the vehicle is full, a new truck is needed
                vehicle_routes.append(route)  ##The complete route is appended to the completed routes by all vehicles
                route = [node_no]  ## start the route again with the last place id
                num_vehicle = num_vehicle + 1  ### next truck
                remained_cap = model.vehicle_cap - model.node_list[node_no].demand  ## update the capacity on new truck
        vehicle_routes.append(route)  ### add last route to completed routes by all vehicles
        return num_vehicle, vehicle_routes

    def calDistance(self, route, model):  ### where the actual distance is calculated
        ### used if we are looking to minimize distance traveled
        distance = 0
        depot = model.depot

        ### The foor loop finds the total distance across all locations with the euclidean distance
        for i in range(len(route) - 1):
            from_node = model.node_list[route[i]]
            to_node = model.node_list[route[i + 1]]
            distance += math.sqrt(
                (from_node.x_coord - to_node.x_coord) ** 2 + (from_node.y_coord - to_node.y_coord) ** 2)
        first_node = model.node_list[route[0]]
        last_node = model.node_list[route[-1]]
        ###Lastly the distance from the depot to the first place and the last place are added
        distance += math.sqrt((depot.x_coord - first_node.x_coord) ** 2 + (depot.y_coord - first_node.y_coord) ** 2)
        distance += math.sqrt((depot.x_coord - last_node.x_coord) ** 2 + (depot.y_coord - last_node.y_coord) ** 2)
        return distance


    def calObj(self, nodes_seq,model):
        num_vehicle, vehicle_routes = self.splitRoutes(nodes_seq, model)

        distance = 0  ## initialize distance
        for route in vehicle_routes:
            distance += self.calDistance(route, model)  ###where the actual distance is calculated
        distTraveled = distance

        if model.opt_type==0:
            obj = num_vehicle
        else:
            obj = distTraveled

        return obj, vehicle_routes, num_vehicle, distTraveled
    def plotObj(self, obj_list):
        plt.rcParams['axes.unicode_minus'] = False   # Show minus sign
        plt.plot(np.arange(1,len(obj_list)+1),obj_list)
        plt.xlabel('Iterations')
        plt.ylabel('Obj Value')
        plt.grid()
        plt.xlim(1,len(obj_list)+1)
        plt.show()
    def outPut(self, model, outFilepath):
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
    def run(self, filepath,Q,alpha,beta,rho,epochs,v_cap,opt_type,popsize, outFilepath):
        """
        :param filepath:Xlsx file path
        :param Q:Total pheromone
        :param alpha:Information heuristic factor
        :param beta:Expected heuristic factor
        :param rho:Information volatilization factor
        :param epochs:Iterations
        :param v_cap: Vehicle capacity
        :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
        :param popsize:Population size
        :return:
        """
        model=Model()
        model.vehicle_cap=v_cap
        model.opt_type=opt_type
        model.alpha=alpha
        model.beta=beta
        model.Q=Q
        model.rho=rho
        model.popsize=popsize
        sol=Sol()
        sol.obj=float('inf')
        model.best_sol=sol
        history_best_obj = []
        self.readXlsxFile(filepath,model)
        self.initParam(model)
        for ep in range(epochs):
            self.movePosition(model)
            self.upateTau(model)
            history_best_obj.append(model.best_sol.obj)
            print("%s/%s， best obj: %s, vehicles: %s, distance: %s" % (ep,epochs, model.best_sol.obj, model.best_sol.numVehicle, model.best_sol.distanceTraveled))
        self.plotObj(history_best_obj)
        self.outPut(model, outFilepath)



