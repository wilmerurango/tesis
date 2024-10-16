# import sys
# import io
# import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gurobipy import *
from itertools import product, combinations
from matplotlib.colors import LinearSegmentedColormap


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

    # # [start] converting demand into behavioural
    # demanda["DemandaComport"] = demanda.apply(lambda fila: clases[fila["Vagon"]][clases[fila["Vagon"]].index(fila["Class"]):][::-1] , axis=1)
    # demanda["DemPotencialTot"] = demanda.apply(behav_demand, axis=1, df=demanda)
    # demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']
    # # [end] converting demand into behavioural

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
    n = len(rota)-1

    # listas de trechos contenidos dentro de otros trechos
    BR = {}
    for i,j in NAD:
        listTemp = list(combinations(rota1[rota1.index(i):rota1.index(j)+1], 2))
        listTemp =  {(ii,jj) for ii,jj in listTemp if (ii,jj) in OD and (ii, jj) != (i,j)} #rota1.index(jj) == rota1.index(ii)+1 and 
        BR[(i,j)] = listTemp

    #indices
    index = [(i,j,v,k,t) for i,j,v,k,t in  indexCombiDem if (i,j) in BR.keys()]
    
    return I, I2, J, OD, NAD, V,  T, stations, VK, P, d, dd, n, BR, index, indexCombiDem, indexCombiDem0, demanda


def create_model(I, rota, VK, NAD, BR, P, Q, d, index, indexCombiDem, indexCombiDem0, demanda, dd):

    model = Model("Modelo 1.1.1")

    # variables de decicion
    X = model.addVars(indexCombiDem0, vtype=GRB.INTEGER , name="X")
    Y = model.addVars(indexCombiDem, vtype=GRB.INTEGER , name="Y")
    A = model.addVars(rota, vtype=GRB.INTEGER , name="A")
    BNA = model.addVars(index, vtype=GRB.BINARY , name="BNA")


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


    for i,j,v,k,t in indexCombiDem:

        VK_ = demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()
    
        pos_k = VK_.index(k)
        last_k = VK_[-1]

        # restricao .5
        model.addConstr(
            X[i,j,v,k,t] <= d[i,j,v,k,t],
            name = f"Assig_({i},{j},{v},{k},{t})"
        )

        # # restricao .demanda comportamental
        # if pos_k >= 1:
        #     model.addConstr(
        #         quicksum(X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= d[i,j,v,k,t],
        #         name = f"DemComp({i},{j},{v},{k},{t})"
        #     )


        if k != last_k:

            # # restricao .5 [1ra parte] modifcacion de la demanda con el porcentaje
            # model.addConstr(
            #     X[i,j,v,k,t] <= d[i,j,v,k,t]*(dd[i,j,v,k,t] / d[i,j,v,last_k,t]),
            #     name = f"Assig1_({i},{j},{v},{k},{t})"
            # )

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
        else:

            # restricao .7
            model.addConstr(
                Y[i,j,v,k,t] >=  X[i,j,v,k,t],
                name=f"Autho_({i},{j},{v},{k},{t})"
            )

            # # restricao .5 [2da parte] modifcacion de la demanda con el porcentaje
            # model.addConstr(
            #     X[i,j,v,k,t] <= dd[i,j,v,k,t] + (d[i,j,v,last_k,t] - quicksum( d[i,j,v,kk,t]*(dd[i,j,v,kk,t] / d[i,j,v,last_k,t]) for kk in VK_)),
            #     name = f"Assig2_({i},{j},{v},{k},{t})"
            # )

        #[start]Restricciones de capitalismo
        if (i,j) in NAD:
            # aqui el (i,j)=(o,d)

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

            for ii,jj,vv,kk,tt in indexCombiDem:
                if (ii,jj) in BR[(i,j)] and v == vv and k == kk and t == tt:
                    # restricao .10.1
                    model.addConstr(
                        Y[ii,jj,vv,kk,tt] <= Q*BNA[i,j,vv,kk,tt],
                        # name = f"pru1({o},{d_},{i},{j},{v},{k},{t})"
                    )

                    # restricao .10.2
                    model.addConstr(
                        BNA[i,j,vv,kk,tt] <= Y[ii,jj,vv,kk,tt],
                        # name = f"pru2({o},{d_},{i},{j},{v},{k},{t})"
                    )
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

    # Optimizar o modelo
    model.optimize()

    return model, A, X, Y, BNA


def save_solution(model, BR, P, d, X, Y, BNA, perio, instance, indexCombiDem):
    print('Valor da função objetivo: ', str(model.ObjVal) )
    print('')
    lista = []
    for i,j, v, k, t in indexCombiDem:
        if (i,j) in BR.keys():
            lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], X[i,j,v,k,t].X, Y[i,j,v,k,t].X , BNA[i,j,v,k,t].X ])
        else:
            lista.append([i+'-'+j,i,j,v, k, t, P[i,j,v,k], d[i,j,v,k,t], X[i,j,v,k,t].X, Y[i,j,v,k,t].X , -1 ])

    a = pd.DataFrame(lista, columns=['o-d',"Origen","Destino",'Vagon','classe','Periodo','Preco','Demanda','Assignments','Authorizations','Binaria'])
    
    # a.to_excel('C:/Users/LAB_C/Documents/wilmer/tesis/ResultadosNew/RestrSemDemanComp/modelo_'+instance+'_t'+str(perio)+'.xlsx', index=False)

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