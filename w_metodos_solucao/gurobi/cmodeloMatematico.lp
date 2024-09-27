\ Model Modelo prueba
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  1915 X[uno,dos,p,0,1] + 869 X[uno,dos,p,1,1] + 72 X[uno,dos,p,2,1]
   + 1849 X[uno,tres,p,0,1] + 1834 X[uno,tres,p,1,1]
   + 231 X[uno,tres,p,2,1] + 1488 X[dos,tres,p,0,1]
   + 1340 X[dos,tres,p,1,1] + 962 X[dos,tres,p,2,1]
Subject To
 Dispo_uno: - A[0] + A[uno] = 0
 Cap_uno: X[uno,dos,p,0,1] + X[uno,dos,p,1,1] + X[uno,dos,p,2,1]
   + X[uno,tres,p,0,1] + X[uno,tres,p,1,1] + X[uno,tres,p,2,1] - A[uno]
   <= 0
 AuthoCap_uno: Y[uno,dos,p,0,1] + Y[uno,tres,p,0,1] <= 100
 Dispo_dos: X[uno,tres,p,0,1] + X[uno,tres,p,1,1] + X[uno,tres,p,2,1]
   - A[uno] + A[dos] = 0
 Cap_dos: X[dos,tres,p,0,1] + X[dos,tres,p,1,1] + X[dos,tres,p,2,1]
   - A[dos] <= 0
 AuthoCap_dos: Y[dos,tres,p,0,1] <= 100
 Assig_(uno,dos,p,0,1): X[uno,dos,p,0,1] <= 36
 Classe_(uno,dos,p,0,1): Y[uno,dos,p,0,1] - Y[uno,dos,p,1,1] >= 0
 Autho_(uno,dos,p,0,1): - X[uno,dos,p,0,1] + Y[uno,dos,p,0,1]
   - Y[uno,dos,p,1,1] >= 0
 Assig_(uno,dos,p,1,1): X[uno,dos,p,1,1] <= 58
 Classe_(uno,dos,p,1,1): Y[uno,dos,p,1,1] - Y[uno,dos,p,2,1] >= 0
 Autho_(uno,dos,p,1,1): - X[uno,dos,p,1,1] + Y[uno,dos,p,1,1]
   - Y[uno,dos,p,2,1] >= 0
 Assig_(uno,dos,p,2,1): X[uno,dos,p,2,1] <= 10
 Autho_(uno,dos,p,2,1): - X[uno,dos,p,2,1] + Y[uno,dos,p,2,1] >= 0
 Assig_(uno,tres,p,0,1): X[uno,tres,p,0,1] <= 12
 Classe_(uno,tres,p,0,1): Y[uno,tres,p,0,1] - Y[uno,tres,p,1,1] >= 0
 Autho_(uno,tres,p,0,1): - X[uno,tres,p,0,1] + Y[uno,tres,p,0,1]
   - Y[uno,tres,p,1,1] >= 0
 Assig_(uno,tres,p,1,1): X[uno,tres,p,1,1] <= 12
 Classe_(uno,tres,p,1,1): Y[uno,tres,p,1,1] - Y[uno,tres,p,2,1] >= 0
 Autho_(uno,tres,p,1,1): - X[uno,tres,p,1,1] + Y[uno,tres,p,1,1]
   - Y[uno,tres,p,2,1] >= 0
 Assig_(uno,tres,p,2,1): X[uno,tres,p,2,1] <= 52
 Autho_(uno,tres,p,2,1): - X[uno,tres,p,2,1] + Y[uno,tres,p,2,1] >= 0
 Assig_(dos,tres,p,0,1): X[dos,tres,p,0,1] <= 68
 Classe_(dos,tres,p,0,1): Y[dos,tres,p,0,1] - Y[dos,tres,p,1,1] >= 0
 Autho_(dos,tres,p,0,1): - X[dos,tres,p,0,1] + Y[dos,tres,p,0,1]
   - Y[dos,tres,p,1,1] >= 0
 Assig_(dos,tres,p,1,1): X[dos,tres,p,1,1] <= 8
 Classe_(dos,tres,p,1,1): Y[dos,tres,p,1,1] - Y[dos,tres,p,2,1] >= 0
 Autho_(dos,tres,p,1,1): - X[dos,tres,p,1,1] + Y[dos,tres,p,1,1]
   - Y[dos,tres,p,2,1] >= 0
 Assig_(dos,tres,p,2,1): X[dos,tres,p,2,1] <= 44
 Autho_(dos,tres,p,2,1): - X[dos,tres,p,2,1] + Y[dos,tres,p,2,1] >= 0
 R30: - Y[uno,tres,p,0,1] + BNA[uno,tres,p,0,1] <= 0
 R31: Y[uno,tres,p,0,1] - 100 BNA[uno,tres,p,0,1] <= 0
 R32: - Y[uno,tres,p,1,1] + BNA[uno,tres,p,1,1] <= 0
 R33: Y[uno,tres,p,1,1] - 100 BNA[uno,tres,p,1,1] <= 0
 R34: - Y[uno,tres,p,2,1] + BNA[uno,tres,p,2,1] <= 0
 R35: Y[uno,tres,p,2,1] - 100 BNA[uno,tres,p,2,1] <= 0
 R36: Y[dos,tres,p,0,1] - 100 BNA[uno,tres,p,0,1] <= 0
 R37: - Y[dos,tres,p,0,1] + BNA[uno,tres,p,0,1] <= 0
 R38: Y[dos,tres,p,1,1] - 100 BNA[uno,tres,p,1,1] <= 0
 R39: - Y[dos,tres,p,1,1] + BNA[uno,tres,p,1,1] <= 0
 R40: Y[dos,tres,p,2,1] - 100 BNA[uno,tres,p,2,1] <= 0
 R41: - Y[dos,tres,p,2,1] + BNA[uno,tres,p,2,1] <= 0
 R42: Y[uno,dos,p,0,1] - 100 BNA[uno,tres,p,0,1] <= 0
 R43: - Y[uno,dos,p,0,1] + BNA[uno,tres,p,0,1] <= 0
 R44: Y[uno,dos,p,1,1] - 100 BNA[uno,tres,p,1,1] <= 0
 R45: - Y[uno,dos,p,1,1] + BNA[uno,tres,p,1,1] <= 0
 R46: Y[uno,dos,p,2,1] - 100 BNA[uno,tres,p,2,1] <= 0
 R47: - Y[uno,dos,p,2,1] + BNA[uno,tres,p,2,1] <= 0
 Inicial_(0,dos,p,0,1): X[0,dos,p,0,1] = 0
 Inicial_(0,dos,p,1,1): X[0,dos,p,1,1] = 0
 Inicial_(0,dos,p,2,1): X[0,dos,p,2,1] = 0
 Inicial_(0,tres,p,0,1): X[0,tres,p,0,1] = 0
 Inicial_(0,tres,p,1,1): X[0,tres,p,1,1] = 0
 Inicial_(0,tres,p,2,1): X[0,tres,p,2,1] = 0
 Cap_0: A[0] = 100
Bounds
Binaries
 BNA[uno,tres,p,0,1] BNA[uno,tres,p,1,1] BNA[uno,tres,p,2,1]
Generals
 X[0,dos,p,0,1] X[0,dos,p,1,1] X[0,dos,p,2,1] X[0,tres,p,0,1]
 X[0,tres,p,1,1] X[0,tres,p,2,1] X[uno,dos,p,0,1] X[uno,dos,p,1,1]
 X[uno,dos,p,2,1] X[uno,tres,p,0,1] X[uno,tres,p,1,1] X[uno,tres,p,2,1]
 X[dos,tres,p,0,1] X[dos,tres,p,1,1] X[dos,tres,p,2,1] Y[uno,dos,p,0,1]
 Y[uno,dos,p,1,1] Y[uno,dos,p,2,1] Y[uno,tres,p,0,1] Y[uno,tres,p,1,1]
 Y[uno,tres,p,2,1] Y[dos,tres,p,0,1] Y[dos,tres,p,1,1] Y[dos,tres,p,2,1]
 A[0] A[uno] A[dos]
End
