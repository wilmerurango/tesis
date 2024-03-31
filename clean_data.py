import pandas as pd
import numpy as np
from itertools import product

def fill_data(oridest, wagons, classes, records_to_correct, periodo=0): #preencha os dados
     
    wagon_class = []
    for v in wagons:
        wagon_class += list(product([v], classes[v]))
        
    try:
        columnas = ['trecho', 'vagon_clase', 'DBD']
        df = pd.DataFrame(list(product(oridest, wagon_class, periodo)), columns=columnas)
    except:
        columnas = ['trecho', 'vagon_clase']
        df = pd.DataFrame(list(product(oridest, wagon_class)), columns=columnas)

    df[['Origin', 'Destination']] = pd.DataFrame(df['trecho'].tolist(), index=df.index)
    df[['Wagon', 'Clase']] = pd.DataFrame(df['vagon_clase'].tolist(), index=df.index)
    df = df.drop(columns=['trecho', 'vagon_clase'])

    try:
        df = df[['Origin', 'Destination', 'Wagon', 'Clase', 'DBD']]
        df['Bookings'] = 0
        df['clave_completa'] = df['Origin'].astype(str) + '_' + df['Destination'].astype(str) + '_' + df['Wagon'].astype(str) + '_' + df['Clase'].astype(str) + '_' + df['DBD'].astype(str)
        records_to_correct['clave_completa'] = records_to_correct['Origin'].astype(str) + '_' + records_to_correct['Destination'].astype(str) + '_' + records_to_correct['Wagon'].astype(str) + '_' + records_to_correct['Clase'].astype(str) + '_' + records_to_correct['DBD'].astype(str)
    except:
        df['Revenue'] = 0
        df['clave_completa'] = df['Origin'].astype(str) + '_' + df['Destination'].astype(str) + '_' + df['Wagon'].astype(str) + '_' + df['Clase'].astype(str)
        records_to_correct['clave_completa'] = records_to_correct['Origin'].astype(str) + '_' + records_to_correct['Destination'].astype(str) + '_' + records_to_correct['Wagon'].astype(str) + '_' + records_to_correct['Clase'].astype(str)

    new_records = df[~df['clave_completa'].isin(records_to_correct['clave_completa'])]
    records_to_correct = pd.concat([records_to_correct, new_records], ignore_index=True)

    records_to_correct.drop('clave_completa', axis=1, inplace=True)
        
    return records_to_correct


def clean_data(route, start_estation, commer_class_hierar):
    # read data
    data = pd.read_csv(route, delimiter=';')
    data.drop(columns=['Unnamed: 0'], inplace=True)
    # data = data[data['DBD']==0]

    # find parameters
    origin = data['Origin'].unique().tolist()
    destin = data['Destination'].unique().tolist()
    periodo = sorted(data['DBD'].unique().tolist(), reverse=True)
    stations = destin + [i for i in origin if i not in destin]

    # anonymize parameters
    anonymo_stat = {i: data[data['Market'] == start_estation + '-' + i]['NbLegs'].unique()[0] + 1 if i != start_estation else 1 for i in stations }
    data.replace(anonymo_stat, inplace=True)
    data['Market'] = data['Origin'].astype('str') +'-'+ data['Destination'].astype('str')
    data['CommercialClass'].replace(commer_class_hierar, inplace=True)

    # sort anonymized parameters
    origin_cor = sorted(data['Origin'].unique().tolist())
    destin_cor = sorted(data['Destination'].unique().tolist())
    oridest = data['Market'].unique().tolist()
    oridest = sorted([(int(i.split('-')[0]), int(i.split('-')[1])) for i in oridest])

    # find parameters
    data[['Wagon','Clase']] = data['CommercialClass'].str.split('-', expand=True).astype('int')
    vagones = data['Wagon'].unique().tolist()
    clases = {v: sorted(data[data['Wagon']==v]['Clase'].unique().tolist()) for v in vagones}

    # Filter demand instance
    demanda = pd.pivot_table(data, values=['Bookings'], index=['Origin','Destination','Wagon','Clase', 'DBD'], aggfunc={'Bookings':'sum'})
    demanda.reset_index(inplace=True)

    # Filter revenue instance
    preco = pd.pivot_table(data, values=['Revenue'], index=['Origin','Destination','Wagon','Clase'], aggfunc={'Revenue':'max'})
    preco.reset_index(inplace=True)

    # fill instance records
    preco = fill_data(oridest,vagones,clases, preco,)         
    demanda = fill_data(oridest,vagones,clases, demanda, periodo)            

    # sort data revenue
    preco = preco.sort_values(by=['Origin', 'Destination', 'Wagon', 'Revenue'], ascending=[True, True, True, False])

    # Transform to dictionary
    dem_cor = demanda.set_index(['Origin', 'Destination', 'Wagon', 'Clase', 'DBD'])['Bookings'].to_dict()
    preco_cor = preco.set_index(['Origin', 'Destination', 'Wagon', 'Clase'])['Revenue'].to_dict()

    return origin_cor, destin_cor, oridest, vagones, periodo, stations, clases, preco_cor, dem_cor, data, anonymo_stat
