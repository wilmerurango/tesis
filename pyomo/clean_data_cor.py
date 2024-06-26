import pandas as pd
import numpy as np
from itertools import product

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
    preco = fill_data(oridest,vagones,clases, preco,)         
    demanda = fill_data(oridest,vagones,clases, demanda, periodo)            

    # sort data revenue
    preco = preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

    # Transform to dictionary
    dem_cor = demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()
    preco_cor = preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()

    return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor
