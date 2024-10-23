# import sys
# import io
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from gurobipy import *
from itertools import product, combinations
from matplotlib.colors import LinearSegmentedColormap
# from collections import defaultdict


def behav_demand(fila, df):
    
    filtro = (df["Origin"] == fila["Origin"]) &  (df["Destination"] == fila["Destination"]) &  (df["Vagon"] == fila["Vagon"]) &  (df["DBD"] == fila["DBD"])
    preferenceList = sorted(df[filtro]["Class"].unique().tolist())
    currentClass = fila["Class"]
    posCurrentClass = preferenceList.index(currentClass)
    sumClass = preferenceList[0:posCurrentClass+1]
    potentialDemand =  df[filtro]
    potentialDemand = potentialDemand[potentialDemand["Class"].isin(sumClass)]

    return  potentialDemand["Bookings"].sum()


def clean_data(demanda, preco):

    demInde = demanda.copy()

    # find parameters
    origin_cor = preco['Origin'].unique().tolist()
    destin_cor = preco['Destination'].unique().tolist()

    oridest = preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
    oridest = oridest.unique().tolist()

    stations = origin_cor + [i for i in destin_cor if i not in origin_cor]

    vagones = preco['Vagon'].unique().tolist()
    clases = {v: sorted(preco[preco['Vagon']==v]['Class'].unique().tolist()) for v in vagones}

    periodo = sorted(demanda['DBD'].unique().tolist(), reverse=True)    

    # [start] converting demand into behavioural
    demanda["DemandaComport"] = demanda.apply(lambda fila: clases[fila["Vagon"]][clases[fila["Vagon"]].index(fila["Class"]):][::-1] , axis=1)
    demanda["DemPotencialTot"] = demanda.apply(behav_demand, axis=1, df=demanda)
    demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']
    # [end] converting demand into behavioural

    # sort data revenue
    preco = preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

    # Transform to dictionary
    dem_cor = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    demInde = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    preco_cor = preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

    #todas las combinaciones de los indices
    # indexCombiPre = [tuple(x) for x in preco[['Origin','Destination','Vagon','Class']].to_numpy()] #para el precio
    indexCombiDem = [tuple(x) for x in demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para l ademanda
    index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, vagones, periodo) for c in clases[v]]
    indexCombiDem0 = indexCombiDem + index_Cero

    return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, demInde, indexCombiDem, indexCombiDem0


def read_data(path_dem:str, path_preco:str, path_rota:str):

    demanda = pd.read_csv(path_dem)
    preco = pd.read_csv(path_preco)
    rota0 = pd.read_csv(path_rota)
    rota1 = eval(rota0['Route'][0])

    return demanda, preco, rota1


# def generar_rutas(origen, destino, nodos):
#     rutas = []
#     idx_origen = nodos.index(origen)
#     idx_destino = nodos.index(destino)

#     # Iniciar la búsqueda de rutas desde el nodo origen
#     def buscar_rutas(ruta_actual, nodo_actual):
#         if nodo_actual == destino:
#             rutas.append(ruta_actual.copy())
#             return

#         # Filtrar nodos válidos que son mayores y están en el rango
#         for siguiente in nodos[idx_origen:idx_destino + 1]:
#             if nodos.index(siguiente) > nodos.index(nodo_actual):  # Solo considerar nodos mayores
#                 ruta_actual.append((nodo_actual, siguiente))
#                 buscar_rutas(ruta_actual, siguiente)
#                 ruta_actual.pop()  # Retroceder para probar otras combinaciones

#     # Comenzar la búsqueda
#     buscar_rutas([], origen)

#     return rutas


# def rotasCombi(BR,stations, OD):
#     CR = {}
#     for o,d in BR.keys():
#         rotas_ = generar_rutas(o,d, stations)

#         pos = []
#         for i in range(len(rotas_)):

#             numOD = len(rotas_[i])
#             cont = 0
#             for ii,jj in rotas_[i]:
#                 if (ii,jj) in OD:
#                     cont += 1

#             if cont != numOD or numOD == 1:
#                 pos.append(rotas_[i])

#         for w in pos:
#             rotas_.remove(w)

#         CR[(o,d)] = rotas_

#     return CR



def find_all_paths_with_tuples(graph_edges, start, end):

    # Crear el grafo a partir de las aristas
    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    
    # Encontrar todas las rutas posibles
    all_paths = list(nx.all_simple_paths(G, source=start, target=end))
    
    # Convertir cada ruta en una lista de tuplas (origen, destino)
    all_paths_with_tuples = []
    for path in all_paths:
        path_tuples = [(path[i], path[i+1]) for i in range(len(path) - 1)]
        all_paths_with_tuples.append(path_tuples)
    
    return all_paths_with_tuples


def create_sets(demanda, preco_, rota1, perio=0):

    if perio != 0:
        periodo_lim = sorted(demanda['DBD'].unique().tolist())[:perio]
        demanda = demanda[demanda['DBD'].isin(periodo_lim)]

    rota = [0] + rota1
    
    I, J, OD, V,  T, stations, VK, P, d, dd, indexCombiDem, indexCombiDem0 = clean_data(demanda, preco_)

    AD = [(i,j) for i,j in OD if rota1.index(j) == rota1.index(i)+1]
    NAD = [(i,j) for i,j in OD if rota1.index(j) != rota1.index(i)+1]

    I = [i for i in rota if i in I]
    I2 = [0] + I# tambem se toma para el para tudo da restriccion de fluxo
    J = [i for i in rota if i in J]
    stations = [i for i in rota if i in stations]
    n = len(rota)-1

    # listas de trechos contenidos dentro de otros trechos
    BR = {}
    for i,j in NAD:
        listTemp = list(combinations(rota1[rota1.index(i):rota1.index(j)+1], 2))
        listTemp =  {(ii,jj) for ii,jj in listTemp if (ii,jj) in OD and (ii, jj) != (i,j)} #rota1.index(jj) == rota1.index(ii)+1 and 
        if len(listTemp) != 0:
            BR[(i,j)] = listTemp

    start_time = time.time()
    
    CR = {}
    # for l,m in BR.keys():
    #     route_ = find_all_paths_with_tuples(BR[l,m], l, m)
    #     CR[(l,m)] = route_
    end_time = time.time()
    tempo = end_time - start_time
    print("demoro: ",tempo)

    #indices
    index = [(i,j,v,k,t) for i,j,v,k,t in  indexCombiDem if (i,j) in BR.keys()]
    
    return I, I2, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, CR, index, indexCombiDem, indexCombiDem0, demanda


def create_model(I, J, rota, VK, NAD, BR, CR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd):

    model = Model("Modelo 1.1.1")

    # variables de decicion
    X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
    Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
    A = model.addVars(rota, vtype=GRB.INTEGER , name="A")
    BNA = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BNA")
    BX = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BX")
    BY = model.addVars(indexCombiDem, vtype=GRB.BINARY , name="BY")


    # funcion objetivo
    model.setObjective(
        quicksum(P[(i,j,v,k)]*X[(i,j,v,k,t)] for i,j,v,k,t in indexCombiDem),
        sense = GRB.MAXIMIZE
    )

    # restricciones
    for i in I:

        # restricao .2
        model.addConstr(
            A[i] == A[rota[rota.index(i)-1]] - 
            quicksum(X[rota[rota.index(i)-1],j,v,k,t] for i_,j,v,k,t in indexCombiDem0 if (rota[rota.index(i)-1] == i_)) + 
            quicksum(X[i_,j,v,k,t] for i_,j,v,k,t in indexCombiDem0 if (j == i)),
            name=f"Dispo_{i}"
        )

        # restricao .3
        model.addConstr(
            quicksum((X[i,j,v,k,t]) for i_,j,v,k,t in indexCombiDem if i == i_) <= A[i],
            name=f"Cap_{i}"
        )

        # restricao .6
        model.addConstr(
            quicksum(Y[i,j,v,VK[v][0],t] for i_,j,v,k,t in indexCombiDem if (i == i_) and (VK[v][0] == k)) <= Q, 
            name=f"AuthoCap_{i}"
        )


    # [start] restricciones Skiplagging
    df = demanda.copy()
    newIndex = df[['Origin', 'Destination', 'Vagon', 'DBD']]
    newIndex = newIndex.drop_duplicates()
    newIndex = [tuple(x) for x in newIndex[['Origin','Destination','Vagon','DBD']].to_numpy()]
    subrota = rota + [w for w in J if w not in rota]

    for i,j,v,t in newIndex:
        VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()

        for ii,jj,vv,tt in newIndex:
            if i == ii and v == vv and t == tt and (subrota.index(jj) > subrota.index(j)): 
                VK2_ = demanda.loc[(demanda["Origin"]==ii) & (demanda["Destination"]==jj) & (demanda["Vagon"]==vv) & (demanda["DBD"]==tt)]["Class"].to_list()

                model.addConstr(
                    quicksum(BY[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BY[ii,jj,vv,k_,t]*P[ii,jj,vv,k_] for k_ in VK2_),
                    name = f"Skiplagging3_({i},{j},{v},{t})"
                )

        # if (i,j) in NAD:
        #     for route in CR[i,j]:
        #         model.addConstr(
        #             quicksum(BY[i,j,v,k_,t]*P[i,j,v,k_] for k_ in VK_) <= quicksum(BY[ii,jj,v,k_,t]*P[ii,jj,v,k_] for ii, jj in route for k_ in VK_ if (ii,jj,v,k_,t) in indexCombiDem),
        #             name = f"Capital_({i},{j},{v},{t})"
        #         )
                
    # [end] restricciones Skiplagging

    for i,j,v,k,t in indexCombiDem:

        VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
        T_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["Class"]==k)]["DBD"].to_list()
        T_ = sorted(T_, reverse=True)
      

        pos_t = T_.index(t)
        pos_k = VK_.index(k)
        last_k = VK_[-1]


        # # restricao .5
        # model.addConstr(
        #     X[i,j,v,k,t] <= d[i,j,v,k,t],
        #     name = f"Assig_({i},{j},{v},{k},{t})"
        # )

        # restricao .demanda comportamental
        if pos_k >= 1:
            model.addConstr(
                quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
                name = f"DemComp({i},{j},{v},{k},{t})"
            )

        # [start] restriccion  fullfillments over periods
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
        # [end] restriccion  fullfillments over periods


        if k != last_k:

            # restricao .5 [1ra parte] modifcacion de la demanda con el porcentaje
            model.addConstr(
                X[i,j,v,k,t] <= d[i,j,v,k,t]*(dd[i,j,v,k,t] / d[i,j,v,last_k,t]),
                name = f"Assig1_({i},{j},{v},{k},{t})"
            )

            # restricao .4
            model.addConstr(
                Y[i,j,v,k,t] >= Y[i,j,v,VK_[pos_k+1],t], 
                name=f"Classe_({i},{j},{v},{k},{t})"
            )

            # # restricao .4.2 [jerarquia para los assigments] =========================================
            # model.addConstr(
            #     X[i,j,v,k,t] <= X[i,j,v,VK_[pos_k+1],t], 
            #     name=f"JerarAssig_({i},{j},{v},{k},{t})"
            # )

            # restricao .8
            model.addConstr(
                Y[i,j,v,k,t] >=  X[i,j,v,k,t] + Y[i,j,v,VK_[pos_k+1],t],
                name=f"Autho_({i},{j},{v},{k},{t})"
            )

            # [start] restricciones Skiplagging
            model.addConstr(
                BY[i,j,v,k,t] >= BNA[i,j,v,k,t] - BNA[i,j,v,VK_[pos_k+1],t],
                name=f"Skiplagging_({i},{j},{v},{k},{t})"
            )

            model.addConstr(
                BY[i,j,v,k,t] <= BNA[i,j,v,k,t] - BNA[i,j,v,VK_[pos_k+1],t],
                name=f"Skiplagging2_({i},{j},{v},{k},{t})"
            )
            # [end] restricciones Skiplagging
        else:

            # restricao .7
            model.addConstr(
                Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                name=f"Autho_({i},{j},{v},{k},{t})"
            )

            # restricao .5 [2da parte] modifcacion de la demanda con el porcentaje
            model.addConstr(
                X[i,j,v,k,t] <= dd[i,j,v,k,t] + (d[i,j,v,last_k,t] - quicksum( d[i,j,v,kk,t]*(dd[i,j,v,kk,t] / d[i,j,v,last_k,t]) for kk in VK_)),
                name = f"Assig2_({i},{j},{v},{k},{t})"
            )

            # [start] restricciones Skiplagging
            model.addConstr(
                BY[i,j,v,k,t] == BNA[i,j,v,k,t]
            )
            # [end] restricciones Skiplagging


        #[start] Restricciones de capitalismo 

        # las restricciones 9.1 y 9.2 antes se hacian solo para los trechos no adjacentes, mas ahora se haran para
        # todos los trechos, esto para agregar las restricciones de Skiplagging.

        # restricao .9.1
        model.addConstr(
            BNA[i,j,v,k,t] <= Y[i,j,v,k,t],
            # name = f"activ_bin_autho_low_({o},{d_},{v},{k},{t})"
        )
        
        # restricao .9.2
        model.addConstr(
            Y[i,j,v,k,t] <= Q*BNA[i,j,v,k,t],
            # name = f"activ_bin_autho_top_({o},{d_},{v},{k},{t})"
            )

        # if (i,j) in NAD:
        #     # aqui el (i,j)=(o,d)
        #     for ii,jj,vv,kk,tt in indexCombiDem:
        #         if (ii,jj) in BR[(i,j)] and v == vv and k == kk and t == tt:
        #             # restricao .10.1
        #             model.addConstr(
        #                 Y[ii,jj,vv,kk,tt] <= Q*BNA[i,j,vv,kk,tt],
        #                 # name = f"pru1({o},{d_},{i},{j},{v},{k},{t})"
        #             )

        #             # restricao .10.2
        #             model.addConstr(
        #                 BNA[i,j,vv,kk,tt] <= Y[ii,jj,vv,kk,tt],
        #                 # name = f"pru2({o},{d_},{i},{j},{v},{k},{t})"
        #             )
        #[end]Restricciones de capitalismo


    for i,j,v,k,t in indexCombiDem0:
        # restricao .11
        if i == 0:
            model.addConstr(
                X[i,j,v,k,t] == 0,
                name = f"Assig_({0},{j},{v},{k},{t})"
            )

    # restricao .12
    model.addConstr(
        A[0] == Q,
        name = f"Cap_0")

    return model, A, X, Y, BNA, BX, BY


def save_solution(model, BR, BX, BY, P, d, X, Y, BNA, perio, instance, indexCombiDem):
    print('Valor da função objetivo: ', str(model.ObjVal) )
    print('')
    lista = []
    for i,j, v, k, t in indexCombiDem:
        # if (i,j) in BR.keys():
        lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], X[i,j,v,k,t].X, Y[i,j,v,k,t].X , BNA[i,j,v,k,t].X, BX[i,j,v,k,t].X, BY[i,j,v,k,t].X])
        # else:
        #     lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], X[i,j,v,k,t].X, Y[i,j,v,k,t].X , -1, BX[i,j,v,k,t].X ])

    a = pd.DataFrame(lista, columns=['o-d',"Origen","Destino",'Vagon','classe','Periodo','Preco','Demanda','Assignments','Authorizations','BNA', 'BX', 'BY'])
    
    a.to_excel('/home/wilmer/Documentos/Codes/tesis/w_metodos_solucao/gurobi/modelo_'+instance+'_t'+str(perio)+'.xlsx', index=False)

    return a


def grafica(a, attrs, h, l, ng, p=-1):
    if p != -1:
        a = a[a['Periodo']==p]

    dfs = []
    for attr in attrs:
        if attr != 'Preco':
            df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','classe'], aggfunc={attr:'sum'})
        else:
            df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','classe'], aggfunc={attr:'max'})

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