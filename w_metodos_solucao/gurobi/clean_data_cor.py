import pandas as pd
import numpy as np
from gurobipy import *
from itertools import product, combinations


def fill_data(oridest, wagons, classes, records_to_correct, periodo=0): #preencha os dados
     
    wagon_class = []
    for v in wagons:
        wagon_class += list(product([v], classes[v]))

    if 'DBD' in records_to_correct.columns:
        columnas = ['trecho', 'vagon_clase', 'DBD']
        df = pd.DataFrame(list(product(oridest, wagon_class, periodo)), columns=columnas)
    else:
        columnas = ['trecho', 'vagon_clase']
        df = pd.DataFrame(list(product(oridest, wagon_class)), columns=columnas)
        
    df[['Origin', 'Destination']] = pd.DataFrame(df['trecho'].tolist(), index=df.index)
    df[['Vagon', 'Class']] = pd.DataFrame(df['vagon_clase'].tolist(), index=df.index)
    df = df.drop(columns=['trecho', 'vagon_clase'])

    if 'DBD' in records_to_correct.columns:
        df = df[['Origin', 'Destination', 'Vagon', 'Class', 'DBD']]
        df['Bookings'] = 0
        df['clave_completa'] = df['Origin'].astype(str) + '_' + df['Destination'].astype(str) + '_' + df['Vagon'].astype(str) + '_' + df['Class'].astype(str) + '_' + df['DBD'].astype(str)
        records_to_correct['clave_completa'] = records_to_correct['Origin'].astype(str) + '_' + records_to_correct['Destination'].astype(str) + '_' + records_to_correct['Vagon'].astype(str) + '_' + records_to_correct['Class'].astype(str) + '_' + records_to_correct['DBD'].astype(str)
    else:
        df['Revenue'] = 0
        df['clave_completa'] = df['Origin'].astype(str) + '_' + df['Destination'].astype(str) + '_' + df['Vagon'].astype(str) + '_' + df['Class'].astype(str)
        records_to_correct['clave_completa'] = records_to_correct['Origin'].astype(str) + '_' + records_to_correct['Destination'].astype(str) + '_' + records_to_correct['Vagon'].astype(str) + '_' + records_to_correct['Class'].astype(str)

    new_records = df[~df['clave_completa'].isin(records_to_correct['clave_completa'])]
    records_to_correct = pd.concat([records_to_correct, new_records], ignore_index=True)

    records_to_correct.drop('clave_completa', axis=1, inplace=True)
        
    return records_to_correct


def behav_demand(fila, df):
    
    filtro = (df["Origin"] == fila["Origin"]) &  (df["Destination"] == fila["Destination"]) &  (df["Vagon"] == fila["Vagon"]) &  (df["DBD"] == fila["DBD"])
    preferenceList = sorted(df[filtro]["Class"].unique().tolist())
    currentClass = fila["Class"]
    posCurrentClass = preferenceList.index(currentClass)
    sumClass = preferenceList[0:posCurrentClass+1]
    potentialDemand =  df[filtro]
    potentialDemand = potentialDemand[potentialDemand["Class"].isin(sumClass)]

    return  potentialDemand["Bookings"].sum()


def clean_data2(demanda, preco):

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
    preco
    # Transform to dictionary
    dem_cor = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    demInde = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    preco_cor = preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

    #todas las combinaciones de los indices
    # indexCombiPre = [tuple(x) for x in preco[['Origin','Destination','Vagon','Class']].to_numpy()] #para el precio
    indexCombiDem = [tuple(x) for x in demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para l ademanda
    index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, vagones, periodo) for c in clases[v]]
    indexCombiDem0 = indexCombiDem + index_Cero

    return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, demInde,indexCombiDem, indexCombiDem0


def clean_data(demanda, preco):
    # find parameters
    origin_cor = preco['Origin'].unique().tolist()
    destin_cor = preco['Destination'].unique().tolist()

    oridest = preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
    oridest = oridest.unique().tolist()

    stations = destin_cor + [i for i in origin_cor if i not in destin_cor]

    vagones = preco['Vagon'].unique().tolist()
    clases = {v: sorted(preco[preco['Vagon']==v]['Class'].unique().tolist()) for v in vagones}

    periodo = sorted(demanda['DBD'].unique().tolist(), reverse=True)

    # fill instance records
    preco = fill_data(oridest,vagones,clases, preco)         
    demanda = fill_data(oridest,vagones,clases, demanda, periodo)       

    # [start] converting demand into behavioural
    demanda["DemandaComport"] = demanda.apply(lambda fila: clases[fila["Vagon"]][clases[fila["Vagon"]].index(fila["Class"]):][::-1] , axis=1)
    demanda["DemPotencialTot"] = demanda.apply(behav_demand, axis=1, df=demanda)
    demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']
    # [end] converting demand into behavioural

    # sort data revenue
    preco = preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

    # Transform to dictionary
    dem_cor = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    preco_cor = preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

    return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor
