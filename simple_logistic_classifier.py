import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import time

np.set_printoptions(threshold=sys.maxsize)


n_rows = 31
n_cols = 36
options = 4

####################################################
#                                                  #
#                   LOAD_IMAGES                    #
#                                                  #
####################################################


def read_imgs(file):
    imagens = []
    labels = []
    arquivo=open(file, "r")
    linha = arquivo.readline()
    while(linha != ''):
        pre_img = linha[:-1].split(' ')
        imagem = np.array([int(float(p)) for p in pre_img[:-1]])
        imagens.append(imagem)
        labels.append(int(pre_img[-1]))
        linha = arquivo.readline()
    return imagens,labels



####################################################
#                                                  #
#      PREPARAÇÃO DAS IMAGENS TREINO & TESTE       #
#                                                  #
####################################################

'''
Input: nº de imagens, imagens, labels
Output: Imagens (85%), Labels (85%), nº de imagens no dataset treino, pesos, erro 
'''
def prep_data_train(N,imagens,labels,n_cols=n_cols,n_rows=n_rows,percentage=0.85):
    Nt= int(N*percentage)
    Ne= int(N*(1-percentage))
    I = int(n_rows*n_cols)
    
    Xt = imagens[:int(N*percentage)]
    Yt = labels[:int(N*percentage)]
    Xe = imagens[int(N*percentage):]
    Ye = labels[int(N*percentage):]

    ew=[x/N for x in np.ones([I+1])]
    err=[]
    err.append(cost(Xt,Yt,Nt,ew))
    print("Iitial error! => ",err)
    return Xt,Yt,Nt,Ne,Xe,Ye,ew,err



####################################################
#                                                  #
#                     PREVISÃO                     #
#                                                  #
####################################################
  
'''
SIGMOID
'''
def sigmoid(s):  
    large=30
    if s<-large: s=-large
    if s>large: s=large
    return (1 / (1 + np.exp(-s)))

'''
BIAS + X * EW[1:]
'''
def predictor(x,ew):
    s=ew[0];
    s=s+np.dot(x,ew[1:])
    sigma=sigmoid(s)
    return sigma

'''
LOSS FUNCTION
'''
def cost(X,Y,N,ew):
    En=0
    epsi=1.e-12
    for n in range(N):
        y=predictor(X[n],ew);
        if y<epsi: y=epsi;
        if y>1-epsi: y=1-epsi;
        En=En+Y[n]*np.log(y)+(1-Y[n])*np.log(1-y)
    En=-En/N
    return En


'''
UPDATE WRONG VALUES
'''
def update(x,y,eta,ew):
    r=predictor(x,ew)
    s=(y-r);
    new_eta=eta
    ew[0]=ew[0]+s*eta
    ew[1:]=ew[1:]+s*eta*x
    return ew

'''
RUN EPOCHS* the prediction + update weights (COM MÉTODO ESTOCÁSTICO)
'''

def run_stocastic(X,Y,N,eta,MAX_ITER,ew,err):
    epsi=0
    it=0
    while(err[-1]>epsi):
        n=int(np.random.rand()*N) # indice aleatório (0-N)
        new_eta=eta
        ew=update(X[n],Y[n],new_eta,ew)
        erro = cost(X,Y,N,ew) 
        err.append(erro)
        print('iter %d, cost=%f, eta=%e     \r' %(it,err[-1],new_eta),end='')
        it=it+1
        if(it>MAX_ITER): break
    return ew, err


####################################################
#                                                  #
#              PLOTABLES & METRICS                 #
#                                                  #
####################################################

def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,25])
    plt.show()
    return 

def confusion(Xeval,Yeval,N,ew):
    C=np.zeros([2,2]);
    for n in range(N):
        y=predictor(Xeval[n],ew)
        if(y<0.5 and Yeval[n]<0.5): C[0,0]=C[0,0]+1; 
        if(y>0.5 and Yeval[n]>0.5): C[1,1]=C[1,1]+1;
        if(y<0.5 and Yeval[n]>0.5): C[1,0]=C[1,0]+1;
        if(y>0.5 and Yeval[n]<0.5): C[0,1]=C[0,1]+1;  
    return C

def accuracy(matrix):
	right_samples = matrix[0,0]+matrix[1,1]
	wrong_samples = matrix[1,0]+matrix[0,1]
	return right_samples/(right_samples+wrong_samples)


####################################################
#                                                  #
#                      CÓDIGO                      #
#                                                  #
####################################################

'''
LOAD IMAGES
'''

imagens,labels = read_imgs('imagens.txt')


'''
SEPARATE DATA INTO TRAIN(Xt) AND TEST(Xe)
'''
start = time.time()
Xt,Yt,Nt,Ne,Xe,Ye,ew,err = prep_data_train(len(imagens),imagens,labels,n_cols,n_rows,0.85)


'''
RUN TRAIN 
'''
# ERROS
#errático -> learn rate muito grande.
#erro parece não mudar ->  learn rate muito baixo
# 5, 4.9,4.7,4.6 ,...(com oscilações) -> lean rate correto


# após o processo posso utilizar os 85%. Posso implementar variações com benchmarking do training e do teste.
# guardar a memória dos learning rates e tipos de transformação etc. -> repetir processo na janela deslizante
# coeficientes podem variar entre tipos de pulling (usar listas de learn rate e iterações na versão final)


ew,err=run_stocastic(Xt,Yt,Nt,0.05,800,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.02,800,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.008,1000,ew,err);print("\n")
ew,err=run_stocastic(Xt,Yt,Nt,0.003,800,ew,err);print("\n")


plot_error(err)

print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))

C = confusion(Xt,Yt,Nt,ew)
print(C)
Acc = accuracy(C)
print(Acc)
print("\n")



'''
RUN TEST
'''

print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
C =confusion(Xe,Ye,Ne,ew)
print(C)
Acc = accuracy(C)
print(Acc)
print("\n")

end = time.time()
print(end - start)