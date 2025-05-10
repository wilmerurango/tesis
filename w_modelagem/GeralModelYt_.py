# cSpell: disable
"""
Esta classe é usada para quando se adiciono o índice de tipos de assentos 
á variável A, e ficou indexada como A[i,v], pelo que esta classe cria um novo 
conjunto chamado IV.

Aleḿ este documento tem um almelhor estructura para a criação dos modelos
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import re

from gurobipy import GRB, Model, quicksum, read
from itertools import product, combinations
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, deque
from pathlib import Path

def method_execut_log(func):
    """Decorador que anota la ejecución de cualquier método de instancia."""
    def wrapper(self, *args, **kwargs):
        #  Ejecutar el método real
        result = func(self, *args, **kwargs)
        #  Registrar que se llamó
        self.execut_log[func.__name__] = \
            self.execut_log.get(func.__name__, 0) + 1
        return result
    return wrapper

class GeralModelYt:
    
    def __init__(self, path_dem, path_preco, path_rota1, Q, perio:int, nome:str, abordagem:str, g_param:dict) -> None:
        
        self.path_dem = Path(path_dem)
        self.path_preco = Path(path_preco)
        self.path_rota1 = Path(path_rota1)
        self.Q = Q
        self.perio = perio
        self.nome = nome
        self.abordagem = abordagem
        self.g_param = g_param

        # Input Data
        self.demanda: pd.DataFrame | None = None
        self.preco: pd.DataFrame | None = None
        self.rota1: list[int] | None = None

        # sets / parámetros.
        self.indexCombiDem = None         #combinacao de todos os indices dos trechos
        self.indexCombiDem0 = None        #combinacao de indices dos trechos usando a estacao zero "0"
        self.indexNoClass = None          #combinacao de indeces sem utilizar as classes (i,j,v,t)
        self.stations = None              #Estacoes da instania
        self.montcar = None               #probabilidade de Montecarlo para as demandas
        self.rota = None                  #Rota
        self.P = None                     #Precos
        self.dl = None                    #demanda Comportamental
        self.d = None                     #demanda Independente
        self.I = None                     #Origens
        self.I2 = None                    #esto é a rota
        self.IV = None
        self.J = None
        self.OD = None
        self.NAD = None
        self.V = None
        self.T = None
        self.VK = None
        self.n = None
        self.BR = None
        self.CR = None
        
        # variaveis gurobi
        self.model: Model | None = None
        self.X = None
        self.Y = None
        self.A = None
        self.BY = None
        self.BX = None
        self.BL = None
        self.BD = None

        # registro de chamadas de metodos
        self.execut_log = {}

        # Salvar Valores das Variaveis num df
        self.result_vars = None

        # Variaves quantidade de restricoes no unimodulares
        self.NamenoUnimodNodelet = None


    def executed(self, nome_):
        """¿El método `nome` se ha ejecutado al menos una vez?"""
        return self.execut_log.get(nome_, 0) > 0

    def save_solution(self, model:str) -> None:

        lista = []
        for i, j, v, k, t in self.indexCombiDem:
            
            if not self.executed('beha_demand_restric'):
                montcar = -1
                demanda = self.d[i,j,v,k,t]
            else:
                demanda = self.dl[i,j,v,k,t]
                montcar = self.montcar[i,j,v,k,t]
                
            lista.append([i+'-'+j,
                            i,
                            j,
                            v, 
                            k, 
                            t, 
                            self.P[i,j,v,k], 
                            demanda, 
                            self.A[i,v].X, 
                            self.X[i,j,v,k,t].X, 
                            self.Y[i,j,v,k,t].X, 
                            montcar, 
                            self.BY[i,j,v,k,t].X, 
                            self.BX[i,j,v,k,t].X, 
                            self.BL[i,j,v,k,t].X, 
                            self.BD[i,j,v,k,t].X]
                        )
        
        result_vars = pd.DataFrame(lista, columns=['o-d',"Origen","Destino",'Vagon','Classe','Periodo','Preco','Demanda', 'AssenVazios[A]', 'Assignments[X]','Authorizations[Y]', 'ProbMontecarlo', '\u03B3', '\u03B1', '\u03B2', '\u03B4'])
        result_vars = result_vars.sort_values(by=["Origen","Destino",'Vagon','Periodo','Classe'])
        self.result_vars = result_vars
        result_vars.to_excel(str(self.path_dem)[:-11] + self.abordagem + '_' + model + '_' + self.nome + '.xlsx', index=False)

    def graph_solution(self) -> None:
    
        if self.g_param['t'] != -1:
            a = self.result_vars[self.result_vars['Periodo']==self.g_param['t']]

        dfs = []
        for attr in self.g_param['attrs']:
            if attr != 'Preco':
                df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','Classe'], aggfunc={attr:'sum'})
            else:
                df = pd.pivot_table(a, values=attr, index='o-d', columns=['Vagon','Classe'], aggfunc={attr:'max'})

            # df.fillna(0, inplace=True)
            dfs.append(df)

        fig, ax = plt.subplots(1, self.g_param['ng'], figsize=(self.g_param['x'], self.g_param['y']))
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
            ax[cont].set_title(self.g_param['attrs'][cont]+ '\n P =' + str(self.g_param['p']))

            cont += 1

        # fig.show()
        # Ajustar márgenes para evitar recortes
        fig.tight_layout(pad=2.0)

        fig.show()

    def non_unimodular_constraint(self, model):
        # Leer el modelo LP
        modelop = read(str(self.path_dem)[:-11] + self.abordagem + '_' + model + '_' + self.nome + '.lp')  # Asegúrate de que esté en el mismo directorio

        # Asegurar que el modelop está actualizado
        modelop.update()

        # Extraer matriz de restricciones
        A = modelop.getA()

        # Extraer nombres de variables (columnas de A)
        variables = [v.VarName for v in modelop.getVars()]

        # Extraer nombres de restricciones (filas de A)
        restricciones = [c.ConstrName for c in modelop.getConstrs()]

        # Extraer lado derecho (b)
        b = [c.RHS for c in modelop.getConstrs()]

        # Convertir la matriz dispersa a DataFrame
        A_df = pd.DataFrame.sparse.from_spmatrix(A, index=restricciones, columns=variables)

        # Convertir b a Series
        # b_series = pd.Series(b, index=restricciones, name='b')


        # print("\nVector b (lado derecho):")
        # print(b_series)

        # print("\nVector x (variables):")
        # print(pd.Series(variables, name='x'))

        # True si el valor ∈ {0, -1, 1}
        en_conjunto = A_df.isin([0, -1, 1])

        # True si TODA la fila está en {0,-1, 1}
        fila_toda_en_conjunto = en_conjunto.all(axis=1)

        # Filas que tienen al menos un valor FUERA del conjunto
        filas_con_valor_distinto = ~fila_toda_en_conjunto

        # Índices que cumplen la condición
        indicesw = A_df.index[filas_con_valor_distinto]

        return indicesw

    def parse_constraints(self, lp_path):
        """
        Lee un archivo .lp y devuelve un conjunto con los nombres de las restricciones
        que aparecen en la sección Subject To (o s.t.).
        """
        constraints = set()
        in_subject_to = False

        # Expresiones regulares para detectar el inicio y final de la sección
        start_pattern = re.compile(r'^(Subject To|s\.t\.|st)\b', re.IGNORECASE)
        end_pattern   = re.compile(r'^(Bounds|Binaries|Generals|End)\b', re.IGNORECASE)

        # Patrón para capturar el nombre de la restricción: texto antes de los dos puntos
        constr_pattern = re.compile(r'^(\S+)\s*:\s*.+')

        with open(lp_path, 'r') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                # ¿Empezamos la sección de restricciones?
                if start_pattern.match(line):
                    in_subject_to = True
                    continue

                # ¿Terminamos la sección?
                if in_subject_to and end_pattern.match(line):
                    break

                # Si estamos dentro de Subject To, intentamos extraer nombre
                if in_subject_to:
                    m = constr_pattern.match(line)
                    if m:
                        name = m.group(1)
                        constraints.add(name)
                    # Si la línea no tiene "nombre:", será continuación de la anterior → la ignoramos

        return constraints

    def find_non_unimod_non_deleted(self, model:str):
        """
        Compara los sets de nombres de restricciones y devuelve la lista
        de las que fueron eliminadas tras el preprocesamiento.
        """
        orig_set = self.parse_constraints(str(self.path_dem)[:-11] + self.abordagem + '_' + model + '_' + self.nome + '.lp')
        pre_set  = self.parse_constraints(str(self.path_dem)[:-11] + self.abordagem + '_' + model + '_' + self.nome + '_pre.lp')

        removed = sorted(orig_set - pre_set)

        non_unimodular = self.non_unimodular_constraint(model)

        non_unimodular_non_deleted = [i for i in non_unimodular if i not in removed]

        # metricas
        percent_non_unimod_non_del = (len(non_unimodular_non_deleted) / self.model.NumConstrs )* 100
        
        print('% Non Unimod. Non Del.: ', round(percent_non_unimod_non_del,2),'%  (', len(non_unimodular_non_deleted),'-', self.model.NumConstrs,')')

        # Variaves quantidade de restricoes no unimodulares
        self.NamenoUnimodNodelet = non_unimodular_non_deleted



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

    @staticmethod
    def montecarlo(fila, merged_df, ns):
        
        class_dem = merged_df.loc[
            (merged_df['Origin']==fila.Origin) & 
            (merged_df['Destination']==fila.Destination) &
            (merged_df['Vagon']==fila.Vagon) &
            (merged_df['DBD']==fila.DBD)]
        
        if class_dem['Class'].shape[0] == 1:
            return 1
        else:
            # Definir os dados da demanda e precos
            precos = class_dem.Revenue.values # precos das classes
            demandas = class_dem.Bookings1.values # demanda independente
            
            # Calcular probabilidades teoricas
            prob_teoricas = [d / demandas.sum() for d in demandas]

            # Gerar a simulacao com escolha aleatoria ponderada
            simulaciones = np.random.choice(precos, size=ns, p=prob_teoricas)

            resultado = np.sum(simulaciones == class_dem.loc[class_dem['Class']==fila.Class].Revenue.values)/ns

            return resultado



    # Preparacoes para o modelo
    def load_raw_data(self) -> None:
        self.demanda = pd.read_csv(self.path_dem)
        self.preco = pd.read_csv(self.path_preco)
        self.rota1 = eval(pd.read_csv(self.path_rota1)["Route"][0])

    def create_sets_parameters(self) -> None:

        # sort data revenue
        preco = self.preco.sort_values(by=['Origin', 'Destination', 'Vagon', 'Revenue'], ascending=[True, True, True, False])

        # sort data demand
        if self.perio:
            periodo_lim = sorted(self.demanda['DBD'].unique().tolist())[:self.perio]
            self.demanda = self.demanda[self.demanda['DBD'].isin(periodo_lim)]

        # Agrupar pelas primeiras 4 colunas e coletar os valores da coluna 5 nas listas
        findClass = self.demanda.groupby(['Origin', 'Destination', 'Vagon', 'DBD'])['Class'].agg(list).reset_index()
        findClass = findClass.set_index(['Origin', 'Destination', 'Vagon', 'DBD'])['Class'].to_dict()

        # find parameters
        rota = [0] + self.rota1
        origens = preco['Origin'].unique().tolist() #origin_cor
        destinations = preco['Destination'].unique().tolist() #destin_cor
        stations = origens + [i for i in destinations if i not in origens]
        oridest = preco[['Origin','Destination']].apply(lambda x: (x['Origin'],x['Destination']), axis=1)
        
        # find conjuntos
        oridest = oridest.unique().tolist()           # Trechos (Origen-Destino)
        I = [i for i in rota if i in origens]         # Estacoes de Origen
        J = [i for i in rota if i in destinations]    # Estacoes de Destino
        I2 = [0] + I                                  # 0 + Estacoes de Origen [também é usado para o "para todo" da restrição de fluxo]
        stations = [i for i in rota if i in stations] # Estacoes
        n = len(rota)-1                               # Quantidade de Estacoes
        V = preco['Vagon'].unique().tolist()          # Vagones
        IV = [(i,v) for i in I2 for v in V]           # Conjunto de (i,v) para o modelo
        T = sorted(self.demanda['DBD'].unique().tolist(), reverse=True)                                       # Periodos
        P = preco.set_index(['Origin', 'Destination', 'Vagon', 'Class'])['Revenue'].to_dict()                 # Preço por trecho
        VK = {v: sorted(preco[preco['Vagon']==v]['Class'].unique().tolist()) for v in V}                      # Classes por vagón
        AD = [(i,j) for i,j in oridest if self.rota1.index(j) == self.rota1.index(i)+1]                       # Trechos adjacentes
        NAD = [(i,j) for i,j in oridest if self.rota1.index(j) != self.rota1.index(i)+1]                      # Trechos não adjacentes
        

        # converting demand into behavioral
        self.demanda["DemandaComport"] = self.demanda.apply(
                                                            lambda fila: self.demanda.loc[
                                                                (self.demanda["Origin"]==fila["Origin"]) & 
                                                                (self.demanda["Destination"]==fila["Destination"]) &
                                                                (self.demanda["Vagon"]==fila["Vagon"]) &
                                                                (self.demanda["DBD"]==fila["DBD"])]["Class"].unique().tolist()[
                                                                    self.demanda.loc[
                                                                    (self.demanda["Origin"]==fila["Origin"]) & 
                                                                    (self.demanda["Destination"]==fila["Destination"]) &
                                                                    (self.demanda["Vagon"]==fila["Vagon"]) &
                                                                    (self.demanda["DBD"]==fila["DBD"])]["Class"].unique().tolist().index(fila["Class"]):
                                                                ][::-1], 
                                                                axis=1
                                                            )
        self.demanda["DemPotencialTot"] = self.demanda.apply(self.behav_demand, axis=1, df=self.demanda)
        self.demanda.columns = ['Origin', 'Destination', 'Vagon', 'Class', 'DBD', "Bookings1", 'PL', 'Bookings']

        # Probalidade de Montecarlo
        merged = self.demanda.merge(preco, on=['Origin', 'Destination', 'Vagon', 'Class'], how='left')
        montecarlo = self.demanda.copy()
        montecarlo['Mtr']= montecarlo.apply(self.montecarlo, axis=1, merged_df=merged, ns=10000)


        # Diccionarios
        dl = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings'].to_dict()  # Demanda Comportamental
        d = self.demanda.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Bookings1'].to_dict()  # Demanda Independente
        montcar = montecarlo.set_index(['Origin', 'Destination', 'Vagon', 'Class', 'DBD'])['Mtr'].to_dict()    # Probabilidade Montecarlo

        # listas de trechos contidos dentro de outros trechos
        BR = {}
        for i,j in NAD:
            listTemp = list(combinations(self.rota1[self.rota1.index(i):self.rota1.index(j)+1], 2))
            listTemp =  {(ii,jj) for ii,jj in listTemp if (ii,jj) in oridest and (ii, jj) != (i,j)} #rota1.index(jj) == rota1.index(ii)+1 and 
            if len(listTemp) != 0:
                BR[(i,j)] = listTemp

        #Todas as possíveis combinações para chegar do origem i até o destino j (Combinated of Routes)
        CR = {}
        for l,m in BR.keys():
            route_ = self.find_all_paths_with_tuples(BR[l,m], l, m)
            CR[(l,m)] = route_
        
        #todas as combinações dos indices
        indexCombiDem = [tuple(x) for x in self.demanda[['Origin','Destination','Vagon','Class','DBD']].to_numpy()] #para a demanda
        index_Cero = [(0, i, v, c, t) for i, v, t in product(stations, V, T) for c in VK[v]]
        indexCombiDem0 = indexCombiDem + index_Cero

        # sets / parámetros.
        self.indexCombiDem = indexCombiDem
        self.indexCombiDem0 = indexCombiDem0
        self.indexNoClass = findClass
        self.stations = stations
        self.montcar = montcar
        self.rota = rota
        self.P = P
        self.dl = dl #demanda Comportamental
        self.d = d  #demanda Independente
        self.I = I
        self.I2 = I2
        self.IV = IV
        self.J = J
        self.OD = oridest
        self.NAD = NAD
        self.V = V
        self.T = T
        self.VK = VK
        self.n = n
        self.BR = BR
        self.CR = CR



# =================================[-START- CREACAO DO MODELO]========================================
    # Criacao das Variaveis do Modelo
    def defini_variables(self) -> None:
        self.X = self.model.addVars(self.indexCombiDem0, vtype=GRB.INTEGER, name="X")
        self.Y = self.model.addVars(self.indexCombiDem, vtype=GRB.INTEGER, name="Y")
        self.A = self.model.addVars(self.IV, vtype=GRB.INTEGER, name="A")              #rota = I2 (antes a rota foi nomeada de I2)
        self.BY = self.model.addVars(self.indexCombiDem, vtype=GRB.BINARY, name="BY")  #Binary all Y           : gamma
        self.BX = self.model.addVars(self.indexCombiDem, vtype=GRB.BINARY, name="BX")  #Binary all X           : alpha
        self.BL = self.model.addVars(self.indexCombiDem, vtype=GRB.BINARY, name="BL")  #Binary Last            : beta
        self.BD = self.model.addVars(self.indexCombiDem, vtype=GRB.BINARY, name="BD")  #Binary exclusive class : Delta



    # Funcoes Objetivo
    @method_execut_log
    def obj_funct_inde(self) -> None:
        self.model.setObjective(
            quicksum(
                self.P[(i,j,v,k)]
                *self.X[(i,j,v,k,t)]
                for i,j,v,k,t in self.indexCombiDem),
            sense = GRB.MAXIMIZE
        )

    def obj_funct_beha(self) -> None:
        self.model.setObjective(
            quicksum(
                self.P[(i,j,v,k)]
                *self.X[(i,j,v,k,t)]
                *self.montcar[(i,j,v,k,t)]
                # *(dd[i,j,v,k,t]/d[i,j,v,demanda.loc[(demanda["Origin"]==i) & (demanda["Destination"]==j) & (demanda["Vagon"]==v) & (demanda["DBD"]==t)]["Class"].to_list()[-1],t])
                for i,j,v,k,t in self.indexCombiDem),
            sense = GRB.MAXIMIZE
        )



    # Restricoes Gerales
    def capacity_restric(self) -> None:
        # restrições de origem
        for i in self.I:
            for v in self.V:
                # restrição .2
                self.model.addConstr(
                    self.A[i,v] == self.A[self.I2[self.I2.index(i)-1],v] - 
                    quicksum(self.X[self.I2[self.I2.index(i)-1],j,v_,k,t] for i_,j,v_,k,t in self.indexCombiDem0 if (self.I2[self.I2.index(i)-1] == i_) and (v_ == v)) + 
                    quicksum(self.X[i_,j,v_,k,t] for i_,j,v_,k,t in self.indexCombiDem0 if (j == i) and (v_ == v)),
                    name=f"A_({i},{v})"
                )

                # restrição .6
                self.model.addConstr(
                    quicksum(self.Y[i_,j,v_,k,t] for i_,j,v_,k,t in self.indexCombiDem if (i == i_) and (v == v_) and  (k == self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()[0]) ) <= self.A[i,v], 
                    name=f"CapY_({i},{v})"
                )

    def relation_x_y_restric(self) -> None:
        for i,j,v,k,t in self.indexCombiDem:

            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            if k != last_k:
                # restrição .8
                self.model.addConstr(
                    self.Y[i,j,v,k,t] >=  self.X[i,j,v,k,t] + self.Y[i,j,v,VK_[pos_k+1],t],
                    name=f"RelXY1_({i},{j},{v},{k},{t})"
                )
            else:
                # restricao .7
                self.model.addConstr(
                    self.Y[i,j,v,k,t] >=  self.X[i,j,v,k,t],
                    name=f"RelXY2_({i},{j},{v},{k},{t})"
                )

    def initial_var_restric(self) -> None:
        for i,j,v,k,t in self.indexCombiDem0:
            # restricao .11
            if i == 0:
                self.model.addConstr(
                    self.X[i,j,v,k,t] == 0,
                    name = f"IniX0_({0},{j},{v},{k},{t})"
                )

        # restricao .12
        for v in self.V:
            self.model.addConstr(
                self.A[0,v] == self.Q[v],
                name = f"IniA0_{v}")
  


    # Restricoes particulares
    def binary_restric(self) -> None:

        if self.executed('obj_funct_inde'):
            dem_temp = self.d
        else:
            dem_temp = self.dl

        for i,j,v,k,t in self.indexCombiDem:

            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
            pos_k = VK_.index(k)
            last_k = VK_[-1]

            # [start] restrição  fulfillments over periods
            self.model.addConstr(
                    self.X[i,j,v,k,t] <= self.BX[i,j,v,k,t] * dem_temp[i,j,v,k,t],
                    name = f"BinBX_u_({i},{j},{v},{k},{t})"
                )
            
            self.model.addConstr(
                self.BX[i,j,v,k,t] <= self.X[i,j,v,k,t],
                name = f"BinBX_l_({i},{j},{v},{k},{t})"
            )
            # [end] restrição  fulfillments over periods


            #[start] restrições de capitalismo 
            # restricao .9.1
            self.model.addConstr(
                self.BY[i,j,v,k,t] <= self.Y[i,j,v,k,t],
                name = f"BinBY_l_({i},{j},{v},{k},{t})"
            )
            
            # restricao .9.2
            self.model.addConstr(
                self.Y[i,j,v,k,t] <= self.Q[v] * self.BY[i,j,v,k,t],
                name = f"BinBY_u_({i},{j},{v},{k},{t})"
                )
            #[end] restrições de capitalismo


            if k != last_k:
                # [start] restrições Skiplagging
                self.model.addConstr(
                    self.BL[i,j,v,k,t] == self.BY[i,j,v,k,t] - self.BY[i,j,v,VK_[pos_k+1],t],
                    name=f"BinBL_f_({i},{j},{v},{k},{t})"
                )
                # [end] restrição Skiplagging
            else:
                # [start] restrições Skiplagging
                self.model.addConstr(
                    self.BL[i,j,v,k,t] == self.BY[i,j,v,k,t],
                    name=f"BinBL_l_({i},{j},{v},{k},{t})"
                )
                # [end] restrições Skiplagging

    def fulfillments_restric(self) -> None:

        for i,j,v,t in self.indexNoClass.keys():

            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
            TC = list(self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v)]["DBD"].unique())
            TC = sorted(TC, reverse=True) # todos os períodos de o origem i, destino j, e Vagon v
            
            # restrição Fulfillment
            if t != TC[0]:
                pos_tc = TC.index(t)
                self.model.addConstr(
                        quicksum(self.BL[i,j,v,k_,t] * self.P[i,j,v,k_] for k_ in VK_) >= quicksum(self.BL[i,j,v,kk,TC[pos_tc-1]] * self.P[i,j,v,kk] for kk in self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==TC[pos_tc-1])]["Class"].to_list() if (i,j,v,kk,TC[pos_tc-1]) in self.indexCombiDem),
                        name = f"FullFill_1_({i},{j},{v},{t})"
                    )

        for i,j,v,k,t in self.indexCombiDem:
            T_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["Class"]==k)]["DBD"].to_list()
            T_ = sorted(T_, reverse=True)
            pos_t = T_.index(t)

            if t != T_[0]:
                self.model.addConstr(
                    self.BX[i,j,v,k,t] <= self.BX[i,j,v,k,T_[pos_t-1]],
                    name = f"FullFill_2_({i},{j},{v},{k},{t})"
                )

    def skiplagging_restric(self) -> None:

        for i,j,v,t in self.indexNoClass.keys():

            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
          
            # os preços das rotas curtas tem que ser menores que os preços das rotas maiores com mesmo origem
            for ii,jj,vv,tt in self.indexNoClass.keys():
                if i == ii and v == vv and t == tt and (self.rota.index(jj) > self.rota.index(j)):
                    VK2_ = self.demanda.loc[(self.demanda["Origin"]==ii) & (self.demanda["Destination"]==jj) & (self.demanda["Vagon"]==vv) & (self.demanda["DBD"]==tt)]["Class"].to_list()

                    self.model.addConstr(
                        quicksum(self.BL[i,j,v,k_,t] * self.P[i,j,v,k_] for k_ in VK_) <= quicksum(self.BL[ii,jj,vv,k_,t] * self.P[ii,jj,vv,k_] for k_ in VK2_),
                        name = f"SkipLag_1_({i},{j},{v},{t})"
                    )
            
            # a suma dos preços das combinações de todas as rotas contidas, tem que ser maiores ou iguais que o preco da rota maior
            if (i,j) in self.BR.keys(): 
                """Esta condicao é feita no BR e nao sob NAN porque: os trechos nao adjacentas do conjunto NAN poden ficar vacios, ou seja sen trechos dentro dele,
                por outro lado, isso nao acontece nas chaves de BR"""

                for route in self.CR[i,j]:

                    listIndex = []
                    cont = 0
                    for ii, jj in route:
                        if (ii, jj, v, t) in self.indexNoClass.keys():
                            cont += 1
                            listatemp = self.indexNoClass[ii, jj, v, t]
                            for k_ in listatemp:
                                listIndex.append((ii,jj,v,k_,t))

                    if len(route) == cont:
                        self.model.addConstr(
                            quicksum(self.BL[i,j,v,k_,t] * self.P[i,j,v,k_] for k_ in VK_) <= quicksum(self.BL[ii,jj,v,k_,t] * self.P[ii,jj,v,k_] for ii, jj, v, k_, t in listIndex),
                            name = f"SkipLag_2_({i},{j},{v},{t})"
                        )

    def demand_x_restric(self) -> None:

        if self.executed('obj_funct_inde'):
            dem_temp = self.d
        else:
            dem_temp = self.dl

        for i,j,v,k,t in self.indexCombiDem:
            self.model.addConstr(
                    self.X[i,j,v,k,t] <= dem_temp[i,j,v,k,t],
                    name = f"DemX_({i},{j},{v},{k},{t})"
                )

    @method_execut_log
    def beha_demand_restric(self) -> None:
        for i,j,v,k,t in self.indexCombiDem:
            
            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
            pos_k = VK_.index(k)

            if pos_k >= 1:
                self.model.addConstr(
                    quicksum(self.X[i,j,v,kk,t] for kk in VK_[0:pos_k+1]) <= self.dl[i,j,v,k,t],
                    name = f"DemBeha_({i},{j},{v},{k},{t})"
                )



    # Melhora do modelo
    def exclusive_class_restric(self) -> None:

        preco_max = max(self.P.values())

        for i,j,v,k,t in self.indexCombiDem:

            VK_ = self.demanda.loc[(self.demanda["Origin"]==i) & (self.demanda["Destination"]==j) & (self.demanda["Vagon"]==v) & (self.demanda["DBD"]==t)]["Class"].to_list()
            pos_k = VK_.index(k)

            if pos_k == 0:
                # [start] Classes Esclusivas
                self.model.addConstr(
                    self.BX[i,j,v,k,t] + quicksum(self.BX[i,j,v,k1,t] for k1 in VK_ if k1 > k ) <= 1 + len(VK_) * (1 - self.BD[i,j,v,k,t]),
                    name = f"ExcluClass_({i},{j},{v},{k},{t})"
                )

                self.model.addConstr(
                    self.P[i,j,v,k] * self.montcar[(i,j,v,k,t)] * self.X[i,j,v,k,t] + (preco_max*self.Q[v])*(1 - self.BD[i,j,v,k,t]) >= quicksum(self.P[i,j,v,k1] * self.montcar[(i,j,v,k,t)] * self.X[i,j,v,k1,t] for k1 in VK_ if k1 > k),
                    name = f"ExcluClass2_({i},{j},{v},{k},{t})"
                    # *(dd[i,j,v,k,t]/d[i,j,v,last_k,t])
                    # *(dd[i,j,v,k1,t]/d[i,j,v,last_k,t])
                )

                self.model.addConstr(
                    self.BX[i,j,v,k,t] <= self.BD[i,j,v,k,t],
                    name = f"ExcluClass3_({i},{j},{v},{k},{t})"
                )
                # [end] Classes Esclusivas

# =================================[-END- CREACAO DO MODELO]========================================



# =================================[-START- EXECUTAR MODELOS]========================================
    # Modelos Base
    def group_inde_base_model(self) -> None:
        self.model = Model(self.nome)
        self.load_raw_data()
        self.create_sets_parameters()
        
        self.defini_variables()
        self.obj_funct_inde()

        self.capacity_restric()
        self.relation_x_y_restric()
        self.initial_var_restric()

    def group_beha_base_model(self) -> None:
        self.model = Model(self.nome)
        self.load_raw_data()
        self.create_sets_parameters()
        
        self.defini_variables()
        self.obj_funct_beha()

        self.capacity_restric()
        self.relation_x_y_restric()
        self.initial_var_restric()

    def run_solver(self, name_model:str):
        preSolM = self.model.presolve()
        preSolM.write(str(self.path_dem)[:-11] + self.abordagem + '_' + name_model + '_' + self.nome + '_pre.lp')
        self.model.write(str(self.path_dem)[:-11] + self.abordagem + '_' + name_model + '_' + self.nome + '.lp')

        start_time_run_model = time.time()
        self.model.optimize()
        end_time_run_model = time.time()

        run_time = end_time_run_model - start_time_run_model

        self.save_solution(name_model)

        if self.model.status == GRB.OPTIMAL:
            self.find_non_unimod_non_deleted(name_model)
        # else:
        #     print("Infactivel")

        return run_time



    # Construcao de modelos independentes
    def independ_base_model(self):

        start_time_crea_model = time.time()
        self.group_inde_base_model()
        self.demand_x_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('independ_base_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def independ_fullfilment_model(self):

        start_time_crea_model = time.time()
        self.group_inde_base_model()
        self.binary_restric()
        self.fulfillments_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('independ_fullfilment_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def independ_skiplagging_model(self):

        start_time_crea_model = time.time()
        self.group_inde_base_model()
        self.binary_restric()
        self.skiplagging_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('independ_skiplagging_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def independ_complete_model(self):

        start_time_crea_model = time.time()
        self.group_inde_base_model()
        self.binary_restric()
        self.fulfillments_restric()
        self.skiplagging_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('independ_complete_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet
        


    # Construcao de modelos comportamentais
    def behavioral_base_model(self):

        start_time_crea_model = time.time()
        self.group_beha_base_model()
        self.demand_x_restric()
        self.beha_demand_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('behavioral_base_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def behavioral_fullfilment_model(self):

        start_time_crea_model = time.time()
        self.group_beha_base_model()
        self.beha_demand_restric()
        self.binary_restric()
        self.fulfillments_restric()
        self.exclusive_class_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('behavioral_fullfilment_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def behavioral_skiplagging_model(self):

        start_time_crea_model = time.time()
        self.group_beha_base_model()
        self.beha_demand_restric()
        self.binary_restric()
        self.skiplagging_restric()
        self.exclusive_class_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('behavioral_skiplagging_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet

    def behavioral_complete_model(self):

        start_time_crea_model = time.time()
        self.group_beha_base_model()
        self.beha_demand_restric()
        self.binary_restric()
        self.fulfillments_restric()
        self.skiplagging_restric()
        self.exclusive_class_restric()
        end_time_crea_model = time.time()

        create_time = end_time_crea_model - start_time_crea_model
        run_time = self.run_solver('behavioral_complete_model')

        return self.model, create_time, run_time, self.NamenoUnimodNodelet
    
# =================================[-END- EXECUTAR MODELOS]========================================
