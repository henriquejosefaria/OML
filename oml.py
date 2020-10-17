import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)


n_rows = 31
n_cols = 36
options = 4

####################################################
#                                                  #
#                   LOAD_IMAGES                    #
#                                                  #
####################################################

def readpgm(name,label):
    with open(name) as f:
        lines = f.readlines()

    # Ignores commented lines
    for l in list(lines):
        if l[0] == '#':
            lines.remove(l)

    # Makes sure it is ASCII format (P2)
    assert lines[0].strip() == 'P2' 

    # Converts data to a list of integers
    data = []

    #print("0 =>",lines[0]) # tipo do ficheiro (P2)
    #print("1 =>",lines[1]) # dimensões do ficheiro (1920,1080)
    #print("2 =>",lines[2]) # valor máximo do ficheiro (255)
    for line in lines[1:]:
        data.extend([int(c) for c in line.split()])
    
    data = (np.array(data[3:]),(data[1],data[0]),data[2])
    data_labels = 25*40*[label]
    data = np.reshape(data[0],data[1])

    return data,data_labels

def remove_inuteis_linha(dados,valor):
  count = len(dados)
  for i in range(len(dados)):
    util = 0
    for x in dados[i]:
      if x != valor:
        util = 1
        break;
    if util == 0:
      dados = np.delete(dados, i, axis=0)
      count -= 1
    if count == i+1:
      return dados
  return dados

def remove_inuteis_col(dados,valor):
  count = len(dados[0])
  for i in range(len(dados[0])):
    util = 0
    if count == i-1:
      return dados
    for x in dados:
      if x[i] != valor:
        util = 1
        break;
    if util == 0:
      dados = np.delete(dados, i, axis=1)
      count -= 1
  return dados

def get_images(data,x=36,y=31):
    images = []
    correction = np.zeros([31,10])
    for i in range(25):
        for j in range(40):
            images.append(data[i*y:(i+1)*y,j*x:(j+1)*x])
    N = len(images)
    data=np.zeros([N,x*y])
    for n in range(N):
        if images[n].shape[0] != 31 or images[n].shape[1] != 36:
            images[n] = np.hstack((images[n], correction))
    for n in range(N):
        for i in range(x*y):
            data[n][i] = images[n][i//x][i%y]
    return np.array(data)

####################################################
#                                                  #
#                  WRITE_IMAGES                    #
#                                                  #
####################################################

def write_imgs(file,imagens,labels):
    arquivo=open(file, "w")
    for i in range(len(imagens)):
        line = ' '.join([str(x) for x in imagens[i]]) + ' ' +str(labels[i])
        arquivo.write(line)
        arquivo.write("\n")



'''
LOAD AND PREPARE IMAGES
'''
data3,labels3 = readpgm('mnist_v5_MNIST-3_00001-01000_25x40.pgm',0)

new_data3 = remove_inuteis_linha(data3,255)
new_data3 = remove_inuteis_col(data3,255)


data8,labels8 = readpgm('mnist_v5_MNIST-8_00001-01000_25x40.pgm',1)

new_data8 = remove_inuteis_linha(data8,255)
new_data8 = remove_inuteis_col(data8,255)


imgs3 = get_images(new_data3[144:919,200:1630])
imgs8 = get_images(new_data8[144:919,200:1630])

imagens = np.concatenate((imgs3,imgs8), axis=0)
labels = labels3 + labels8

'''
SHUFFLE IMAGES
'''
idx = np.random.permutation(len(imagens))
imagens = np.array([imagens[i] for i in idx])
labels = [labels[i] for i in idx]

write_imgs('imagens.txt',imagens,labels)


'''
SEPARATE DATA INTO TRAIN(Xt) AND TEST(Xe)
'''

#Xt,Yt,Nt,Ne,Xe,Ye,ew,err = prep_data_train(len(imagens),imagens,labels,n_cols,n_rows,0.2)


'''
RUN TRAIN 
'''
# ERROS
#errático -> learn rate muito grande.
#erro parece não mudar ->  learn rate muito baixo
# 5, 4.9,4.7,4.6 ,...(com oscilações) -> lean rate correto

#quantas iterações com taxa de aprendizagem
# utilizar 20% dos dados por questões de rapidez!!!
# 5,4 ,3.7,...,3.7,3.6,3.5,...,3,7,3,6,.. -> qual a iteração com que chego a 3.7 (será o número de iterações) 
#              -> voltar a chamar com taxa de aprendizagem mais baixa (1/3 ou 1/4 - normalmente )
# como reduzir -> prep_data_Train com 15% por exemplo
# nº de chamadas run_stocastic varia, pode ser mais ou menos! - parar quando não acontece mais nada

# após o processo posso utilizar os 85%. Posso implementar variações com benchmarking do training e do teste.
# guardar a memória dos learning rate e tipos de transformação etc. -> repetir processo na janela deslizante
# coeficientes podem variar entre tipos de pulling (usar listas de learn rate e iterações na versão final)


#(label = 0/1)
'''
ew,err=run_stocastic(Xt,Yt,Nt,0.33,12,ew,err);print("\n")     # 12
ew,err=run_stocastic(Xt,Yt,Nt,0.11,8,ew,err);print("\n")      # 20
ew,err=run_stocastic(Xt,Yt,Nt,0.0275,15,ew,err);print("\n")   # 35
ew,err=run_stocastic(Xt,Yt,Nt,0.0092,15,ew,err);print("\n")   # 50
ew,err=run_stocastic(Xt,Yt,Nt,0.001,17,ew,err);print("\n")    # 67
ew,err=run_stocastic(Xt,Yt,Nt,0.00006,13,ew,err);print("\n")  # 80
ew,err=run_stocastic(Xt,Yt,Nt,0.00001,200,ew,err);print("\n") #
#ew,err=run_stocastic(Xt,Yt,Nt,0.0275,200,ew,err);print("\n")

'''



#plot_error(err)

#print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))
#C = confusion(Xt,Yt,Nt,ew)
#print(C)

'''
RUN TEST
'''

#print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
#C =confusion(Xe,Ye,Ne,ew)
#print(C)



