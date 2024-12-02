# type: ignore
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *
from itertools import product, combinations
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, deque

# MODELOS BASE
class BaseModel:

    def __init__(self, path_dem, path_preco, path_rota1, Q, perio):
        self.demanda = pd.read_csv(path_dem)
        self.preco = pd.read_csv(path_preco)
        self.rota1 = eval(pd.read_csv(path_rota1)['Route'][0])
        self.Q = Q
        self.perio = perio

    def clean_data(self):
        # find parameters
        origin_cor = self.preco['Origin'].unique().tolist()
        destin_cor = self.preco['Destination'].unique().tolist()

        oridest = self.preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
        oridest = oridest.unique().tolist()

        stations = origin_cor + [i for i in destin_cor if i not in origin_cor]

        vagones = self.preco['Vagon'].unique().tolist()
        clases = {v: sorted(self.preco[self.preco['Vagon']==v]['Class'].unique().tolist()) for v in vagones}

        periodo = sorted(self.demanda['DBD'].unique().tolist(), reverse=True)    

        # sort data revenue
        self.preco = self.preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

        # Transform to dictionary
        dem_cor = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        demInde = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        preco_cor = self.preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

        #todas as combinações dos indices
        # indexCombiPre = [tuple(x) for x in preco[['Origin','Destination','Vagon','Class']].to_numpy()] #para o preço
        indexCombiDem = [tuple(x) for x in self.demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para a demanda
        index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, vagones, periodo) for c in clases[v]]
        indexCombiDem0 = indexCombiDem + index_Cero

        return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, demInde, indexCombiDem, indexCombiDem0

    def create_sets(self):

        if self.perio != 0:
            periodo_lim = sorted(self.demanda['DBD'].unique().tolist())[:self.perio]
            self.demanda = self.demanda[self.demanda['DBD'].isin(periodo_lim)]

        rota = [0] + self.rota1
        
        I, J, OD, V,  T, stations, VK, P, d, dd, indexCombiDem, indexCombiDem0 = self.clean_data()

        AD = [(i,j) for i,j in OD if self.rota1.index(j) == self.rota1.index(i)+1]
        NAD = [(i,j) for i,j in OD if self.rota1.index(j) != self.rota1.index(i)+1]

        I = [i for i in rota if i in I]
        I2 = [0] + I    #também é usado para o "para todo" da restrição de fluxo
        J = [i for i in rota if i in J]
        stations = [i for i in rota if i in stations]
        n = len(rota)-1

        # listas de trechos contidos dentro de outros trechos
        BR = {}
        for i,j in NAD:
            listTemp = list(combinations(self.rota1[self.rota1.index(i):self.rota1.index(j)+1], 2))
            listTemp =  {(ii,jj) for ii,jj in listTemp if (ii,jj) in OD and (ii, jj) != (i,j)} #rota1.index(jj) == rota1.index(ii)+1 and 
            if len(listTemp) != 0:
                BR[(i,j)] = listTemp
        
        #indices
        index = [(i,j,v,k,t) for i,j,v,k,t in  indexCombiDem if (i,j) in BR.keys()]

        # Agrupar pelas primeiras 4 colunas e coletar os valores da coluna 5 nas listas
        findClass = self.demanda.groupby(['Origin', 'Destination', 'Vagon', 'DBD'])['Class'].agg(list).reset_index()
        findClass = findClass.set_index(['Origin', 'Destination', 'Vagon', 'DBD'])['Class'].to_dict()

        #criar conjunto iv
        IV = [(i,v) for i in I2 for v in V]
        
        #Todas as possíveis combinações para chegar do origem i até o destino j (Combinated of Routes)
        CR = {}
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, self.demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()
       
        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()

            pos_k = VK_.index(k)
            last_k = VK_[-1]


            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )


            if k != last_k:

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class HierarBehavioralModel(BaseModel):
    
    @staticmethod
    def behav_demand(fila, df):
        filtro = (df["Origin"] == fila["Origin"]) &  (df["Destination"] == fila["Destination"]) &  (df["Vagon"] == fila["Vagon"]) &  (df["DBD"] == fila["DBD"])
        preferenceList = sorted(df[filtro]["Class"].unique().tolist())
        currentClass = fila["Class"]
        posCurrentClass = preferenceList.index(currentClass)
        sumClass = preferenceList[0:posCurrentClass+1]
        potentialDemand =  df[filtro]
        potentialDemand = potentialDemand[potentialDemand["Class"].isin(sumClass)]

        return  potentialDemand["Bookings"].sum()

    def clean_data(self):
        # find parameters
        origin_cor = self.preco['Origin'].unique().tolist()
        destin_cor = self.preco['Destination'].unique().tolist()

        oridest = self.preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
        oridest = oridest.unique().tolist()

        stations = origin_cor + [i for i in destin_cor if i not in origin_cor]

        vagones = self.preco['Vagon'].unique().tolist()
        clases = {v: sorted(self.preco[self.preco['Vagon']==v]['Class'].unique().tolist()) for v in vagones}

        periodo = sorted(self.demanda['DBD'].unique().tolist(), reverse=True)    

        # [start] converting demand into behavioral
        self.demanda["DemandaComport"] = self.demanda.apply(lambda fila: clases[fila["Vagon"]][clases[fila["Vagon"]].index(fila["Class"]):][::-1] , axis=1)
        self.demanda["DemPotencialTot"] = self.demanda.apply(self.behav_demand, axis=1, df=self.demanda)
        self.demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']
        # [end] converting demand into behavioral

        # sort data revenue
        self.preco = self.preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

        # Transform to dictionary
        dem_cor = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        demInde = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        preco_cor = self.preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

        #todas as combinações dos indices
        # indexCombiPre = [tuple(x) for x in preco[['Origin','Destination','Vagon','Class']].to_numpy()] #para o preço
        indexCombiDem = [tuple(x) for x in self.demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para a demanda
        index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, vagones, periodo) for c in clases[v]]
        indexCombiDem0 = indexCombiDem + index_Cero

        return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, demInde, indexCombiDem, indexCombiDem0

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )

        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()

            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            if k != last_k:

                # restrição .4.2 [hierarquia para os assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class PercentBehavioralModel(BaseModel):
    
    @staticmethod
    def behav_demand(fila, df):
        filtro = (df["Origin"] == fila["Origin"]) &  (df["Destination"] == fila["Destination"]) &  (df["Vagon"] == fila["Vagon"]) &  (df["DBD"] == fila["DBD"])
        preferenceList = sorted(df[filtro]["Class"].unique().tolist())
        currentClass = fila["Class"]
        posCurrentClass = preferenceList.index(currentClass)
        sumClass = preferenceList[0:posCurrentClass+1]
        potentialDemand =  df[filtro]
        potentialDemand = potentialDemand[potentialDemand["Class"].isin(sumClass)]

        return  potentialDemand["Bookings"].sum()

    def clean_data(self):
        # find parameters
        origin_cor = self.preco['Origin'].unique().tolist()
        destin_cor = self.preco['Destination'].unique().tolist()

        oridest = self.preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
        oridest = oridest.unique().tolist()

        stations = origin_cor + [i for i in destin_cor if i not in origin_cor]

        vagones = self.preco['Vagon'].unique().tolist()
        clases = {v: sorted(self.preco[self.preco['Vagon']==v]['Class'].unique().tolist()) for v in vagones}

        periodo = sorted(self.demanda['DBD'].unique().tolist(), reverse=True)    

        # [start] converting demand into behavioral
        self.demanda["DemandaComport"] = self.demanda.apply(lambda fila: clases[fila["Vagon"]][clases[fila["Vagon"]].index(fila["Class"]):][::-1] , axis=1)
        self.demanda["DemPotencialTot"] = self.demanda.apply(self.behav_demand, axis=1, df=self.demanda)
        self.demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']
        # [end] converting demand into behavioral

        # sort data revenue
        self.preco = self.preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

        # Transform to dictionary
        dem_cor = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        demInde = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
        preco_cor = self.preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

        #todas as combinações dos indices
        # indexCombiPre = [tuple(x) for x in preco[['Origin','Destination','Vagon','Class']].to_numpy()] #para o preço
        indexCombiDem = [tuple(x) for x in self.demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para a demanda
        index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, vagones, periodo) for c in clases[v]]
        indexCombiDem0 = indexCombiDem + index_Cero

        return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, demInde, indexCombiDem, indexCombiDem0

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )

        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()

            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            if k != last_k:

                # restrição .5 [1ra parte] modificação de la demanda con el porcentagem
                model.addConstr(
                    X[i,j,v,k,t] <= d[i,j,v,k,t]*(dd[i,j,v,k,t] / d[i,j,v,last_k,t]),
                    name = f"Assig1_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # restricao .5 [2da parte] modificação da demanda con a porcentagem
                model.addConstr(
                    X[i,j,v,k,t] <= dd[i,j,v,k,t] + (d[i,j,v,last_k,t] - quicksum( d[i,j,v,kk,t]*(dd[i,j,v,kk,t] / d[i,j,v,last_k,t]) for kk in VK_)),
                    name = f"Assig2_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem



# MODELOS INDEPENDENTES
class BaseModel_Fulfillments(BaseModel):
    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Fulfillments
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v

            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Fulfillments


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]


            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods


            if k != last_k:

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class BaseModel_Skiplagging(BaseModel):
    
    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]


            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            if k != last_k:

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging

            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class BaseModel_Fulfillments_Skiplagging(BaseModel):

    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[(l,m)], l, m)
            CR[(l,m)] = route_

        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")
 
        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),# + 
            # quicksum(P[(i,j,v,k)]*Y[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )

            # restrição Fulfillment
            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True) # todos os períodos de o origem i, destino j, Vagon v e classe k

            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]


            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods
            
            
                

            if k != last_k:

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )

                # model.addConstr(
                #     BL[i,j,v,k,t] <= BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                #     name=f"Skiplagging2_({i},{j},{v},{k},{t})"
                # )
                # [end] restrição Skiplagging

            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem



# MODELOS COMPORTAMENTAIS [Demanda tipo Hierarquia]
class HierarBehavioralModel_Fulfillments(HierarBehavioralModel):
    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )


        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )

        # [start] restrições Fulfillment
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
           
            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Fulfillment


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)
           
            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class HierarBehavioralModel_Skiplagging(HierarBehavioralModel):

    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class HierarBehavioralModel_Fulfillments_Skiplagging(HierarBehavioralModel):

    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )

            # restrição Fulfillment
            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem



# MODELOS COMPORTAMENTAIS [Demanda tipo Percentages]
class PercentBehavioralModel_Fulfillments(PercentBehavioralModel):
    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )


        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Fulfillment
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Fulfillment


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)
           
            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class PercentBehavioralModel_Skiplagging(PercentBehavioralModel):

    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )


        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem

class PercentBehavioralModel_Fulfillments_Skiplagging(PercentBehavioralModel):

    @staticmethod
    def find_all_paths_with_tuples(tuplas, start, end):

        # Criar um dicionario de adjacência pelas conexiones
        grafo = defaultdict(list)
        for o, d in tuplas:
            grafo[o].append(d)
        
        # Se o destino nao está conectado com nenhum nó, nao ha rotas possíveis
        if end not in grafo and all(end != d for _, d in tuplas):
            return []
        
        # Inicializar a fila para BFS
        rutas = []
        cola = deque([(start, [start])])  # (nodo actual, ruta acumulada)
        
        while cola:
            nodo_actual, ruta_actual = cola.popleft()
            
            # Se chegamos ao destino, transformar a rota em tuplas e salvar
            if nodo_actual == end:
                rutas.append([(ruta_actual[i], ruta_actual[i+1]) for i in range(len(ruta_actual) - 1)])
                continue
            
            # Explorar los vizinhos del nodo actual
            for vizinho in grafo[nodo_actual]:
                if vizinho not in ruta_actual:  # Evitar ciclos
                    cola.append((vizinho, ruta_actual + [vizinho]))
        
        return rutas
    
    def create_sets(self):

        I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = super().create_sets()

        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        return I, I2, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass

    def create_model(self): #I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd
        
        I, rota, IV, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda, findClass = self.create_sets()

        model = Model("Modelo 1.1.1")

        # variables de decision
        X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
        Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
        A = model.addVars(IV, vtype=GRB.INTEGER , name="A") #rota = I2 (antes a rota foi nomeada de I2)
        BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")  #Binary all Y
        BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")  #Binary all X
        BL = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BL")  #Binary Last


        # função objetivo
        model.setObjective(
            quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
            sense = GRB.MAXIMIZE
        )

        # restrições de origem
        for i in I:
            for v in V:
                # restrição .2
                model.addConstr(
                    A[i,v] == A[rota[rota.index(i)-1],v] - 
                    quicksum(X[rota[rota.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(X[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"Dispo_{i,v}"
                )

                # restrição .3
                model.addConstr(
                    quicksum((X[i_,j,v_,k,t]) for i_,j,v_,k,t in indexCombiDem if (i_ == i) and (v_ == v)) <= A[i,v],
                    name=f"Cap_{i,v}"
                )

                # restrição .6
                model.addConstr(
                    quicksum(Y[i_,j,v_,k,t] for i_,j,v_,k,t in indexCombiDem if (i == i_) and (v == v_) and  (k == demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.Q[v], 
                    name=f"AuthoCap_{i,v}"
                )


        # [start] restrições Skiplagging
        df = demanda.copy()
        newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
        newIndex = newIndex.drop_duplicates()
        newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
        subrota = rota + [w for w in J if w not in rota]

        for i,j,v,t in newIndex:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            TC = list(demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in newIndex:
                if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)):
                    VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                    model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"Skiplagging3_({i},{j},{v},{t})"
                    )
            

            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in BR.keys(): # if (i,j) in NAD:

                for route in CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in findClass:
                            cont += 1
                            listatemp = findClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

 
                    if len(route) == cont:
                        model.addConstr(
                            quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BL[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"Capital_({i},{j},{v},{t})"
                        )

            # restrição Fulfillment
            if t != TC[0]:
                pos_tc = TC.index(t)
                model.addConstr(
                        quicksum(BL[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) >= quicksum(BL[i,j,v,kk,TC[pos_tc-1]]*P[i,j,v,kk] for kk in demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in indexCombiDem),
                        name = f"PrecoTemAs_({i},{j},{v},{t})"
                    )
        # [end] restrições Skiplagging


        for i,j,v,k,t in indexCombiDem:

            VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
            T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)

            
            pos_t = T_.index(t)
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # restrição .5
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t],
                name = f"Assig_({i},{j},{v},{k},{t})"
            )

            # restrição .demanda comportamental
            if pos_k >= 1:
                model.addConstr(
                    quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                    name = f"DemComp({i},{j},{v},{k},{t})"
                )

            # [start] restrição  fulfillments over periods
            model.addConstr(
                X[i,j,v,k,t] <= BX[i,j,v,k,t]*d[i,j,v,k,t],
                name = f"FullPeriod_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BX[i,j,v,k,t] <= X[i,j,v,k,t],
                name = f"FullPeriod2_({i},{j},{v},{k},{t})"
            )

            if t != T_[0]:
                model.addConstr(
                    # BX[i,j,v,k,t] <= quicksum(BX[i,j,v,k,t_] for t_ in T_[0:pos_t]),
                    BX[i,j,v,k,t] <= BX[i,j,v,k,T_[pos_t-1]],
                    name = f"BinX({i},{j},{v},{k},{t})"
                )
            # [end] restrição  fulfillments over periods

            if k != last_k:

                # restrição .4.2 [jerarquia para los assignments] =========================================
                model.addConstr(
                    X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
                    name=f"JerarAssig_({i},{j},{v},{k},{t})"
                )

                # restrição .8
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t] - BY[i,j,v,VK_[pos_k+1],t],
                    name=f"Skiplagging_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging
            else:

                # restricao .7
                model.addConstr(
                    Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                    name=f"Autho_({i},{j},{v},{k},{t})"
                )

                # [start] restrições Skiplagging
                model.addConstr(
                    BL[i,j,v,k,t] == BY[i,j,v,k,t]
                )
                # [end] restrições Skiplagging


            #[start] restrições de capitalismo 
            # as restrições 9.1 e 9.2 antes se faziam só para os trechos nao adjacentes, mas agora se faram para
            # todos os trechos, esto para adicionar as restrições de Skiplagging.

            # restricao .9.1
            model.addConstr(
                BY[i,j,v,k,t] <= Y[i,j,v,k,t],
                # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
            )
            
            # restricao .9.2
            model.addConstr(
                Y[i,j,v,k,t] <= self.Q[v]*BY[i,j,v,k,t],
                # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


        for i,j,v,k,t in indexCombiDem0:
            # restricao .11
            if i == 0:
                model.addConstr(
                    X[i,j,v,k,t] == 0,
                    name = f"Assig_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in V:
            model.addConstr(
                A[0,v] == self.Q[v],
                name = f"Cap_0_{v}")

        return model, A, X, Y, BY, BX, BL, P, d, self.perio, indexCombiDem



# FUNÇÕES ADICIONAIS
def save_solution(model, BX, BL, P, d, X, Y, A, BY, perio, nameModel, indexCombiDem, path):

    lista = []
    for i, j, v, k, t in indexCombiDem:
        # if (i,j) in BR.keys():
        lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], A[i,v].X, X[i,j,v,k,t].X, Y[i,j,v,k,t].X , BY[i,j,v,k,t].X, BX[i,j,v,k,t].X, BL[i,j,v,k,t].X])
        # else:
        #     lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], X[i,j,v,k,t].X, Y[i,j,v,k,t].X , -1, BX[i,j,v,k,t].X ])

    a = pd.DataFrame(lista, columns=['o-d',"Origen","Destino",'Vagon','Classe','Periodo','Preco','Demanda', 'AssenVazios[A]', 'Assignments[X]','Authorizations[Y]','BY', 'BX', 'BL'])
    
    a.to_excel(path+'/'+ nameModel +'.xlsx', index=False)

    return a

def grafica(a, attrs, h, l, ng, p=-1):
    if p != -1:
        a = a[a['Periodo']==p]

    dfs = []
    for attr in attrs:
        if attr != 'Preco':
            df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','Classe'], aggfunc={attr:'sum'})
        else:
            df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','Classe'], aggfunc={attr:'max'})

        # df.fillna(0, inplace=True)
        dfs.append(df)

    fig, ax = plt.subplots(1, ng, figsize=(l, h)) 
    # plt.figure(figsize=(4, 3))  # Ajusta el ancho a 8 y la altura a 6

    colores = [(0, 'red'), (0.1, 'yellow'), (0.75, 'orange'), (1, 'green')] 
    cmap_customi = LinearSegmentedColormap.from_list('mi_colormap', colores)

    cont = 0
    for df in dfs:
        # Agregar una escala de color usando pcolor
        ax[cont].pcolor(df, cmap=cmap_customi, edgecolors='w', linewidths=2)
        # plt.colorbar(ax[cont].pcolor(df, cmap=cmap_customi, edgecolors='w', linewidths=2))

        # Añadir etiquetas de los ejes
        # if cont == 0:
        ax[cont].set_yticks(np.arange(0.5, len(df.index)), df.index)
        ax[cont].set_xticks(np.arange(0.5, len(df.columns)), df.columns)

        ax[cont].xaxis.set_ticks_position('top')
        ax[cont].tick_params(axis='x', rotation=45, labelsize=8)

        # Agregar valores numéricos a cada celda
        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                if pd.isna(df.iloc[i, j]):
                    valor = "-"
                else:
                    valor = int(df.iloc[i, j])
                ax[cont].text(j + 0.5, i + 0.5, str(valor), color='black', ha='center', va='center')

        # Añadir título y etiquetas
        ax[cont].set_title(attrs[cont])

        cont += 1

    # fig.show()
    # Ajustar márgenes para evitar recortes
    fig.tight_layout(pad=2.0)

    return fig

