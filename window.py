import numpy as np
import matplotlib.pyplot as plt
import sys
import time

np.set_printoptions(threshold=sys.maxsize)


n_rows = 31
n_cols = 36
options = 8

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
    print("Initial error! => ",err)
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
#                    PLOTABLES                     #
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
#                REDUÇÃO DE IMAGENS                #
#                                                  #
####################################################


#pooling classico!
def max_func(data): 
    return max(data)

def min_func(data):
    return min(data)

def average_func(data):
    return sum(data)/len(data)

def max_centered_func(data):
    return max_func(data)-average_func(data)

'''

# 3 

 xxxxxxxx
       xx
       xx
    xxxxx
       xx
       xx
 xxxxxxxx

# 8  

xxxxxxxxx
xx     xx
xx     xx
xxxxxxxxx
xx     xx
xx     xx
xxxxxxxxx

# 3
      xxxx
   xxx    xx
           xx
         xxxx
       xxx   xx
              xx
             xxx
          xxxx

# 8
      xxxx
   xxx    xx
   xx      xx
   xx    xxxx
     xxxxx   xx
      xx      xx
       xx     xxx
        xxxxxxx
          

# FILTROS

#########  <-
# x x   #  <-
#   x   #  <-
#   x x #  <-
#########  <-
#########
# x x   #
#   x x #
#########
#######
# x   #
# x x #
#   x #
#######
#######
# x x #
#   x #
#######
'''



#poolling muito grande (6x8) e fazer stride de 4x5/3x4. -> verificar a falha na identificação dos 3 e 8 (usar a melhor função de poolling)


#pooling extra! (exóticas)
def diag_plus_vert_avg(data,size_x,size_y):  
    lista  = []
    pos    = 0
    center = int(size_x/2)
    while(size_y > pos and size_x > pos):
        if (pos + pos * size_x) == (center + pos * size_x):
            lista.append(data[pos + pos * size_x])
        else:
            lista.append(data[pos + pos * size_x])
            lista.append(data[center + pos*size_x])
        pos = pos + 1
    return sum(lista)/len(lista)

def diag_plus_vert_max_centered(data,size_x,size_y):
    lista  = []
    pos    = 0
    center = int(size_x/2)
    while(size_y > pos and size_x > pos):
        if (pos + pos * size_x) == (center + pos * size_x):
            lista.append(data[pos + pos * size_x])
        else:
            lista.append(data[pos + pos * size_x])
            lista.append(data[center + pos*size_x])
        pos = pos + 1
    return max(lista) - sum(lista)/len(lista)
    
# min das diagonais e da vertical e aplicar o max dos 3; 
def diag_vert_max(data,size_x,size_y):
    diag1 = []
    diag2 = []
    vert  = []
    pos = 0
    center = int(size_x/2)
    while(size_y > pos and size_x > pos):
        diag1.append(data[pos*size_x + pos])
        diag2.append(data[(pos+1)*size_x - (pos+1)])
        vert.append(data[center+pos*size_x])
        pos = pos + 1
    return max(min(diag1),max(min(diag2),min(vert)))

# fazer o average para as diagonais e vertical e obter o min;
def diag_vert_min_avg(data,size_x,size_y):
    diag1 = []
    diag2 = []
    vert  = []
    pos = 0
    center = int(size_x/2)
    while(size_y > pos and size_x > pos):
        diag1.append(data[pos*size_x + pos])
        diag2.append(data[(pos+1)*size_x - (pos+1)])
        vert.append(data[center+pos*size_x])
        pos = pos + 1
    return min(sum(diag1)/size_y,min(sum(diag2)/size_y,sum(vert)/size_y))



#pooling para compressão máxima sem perder classificação

def window_func(data,option,size_x,size_y):
    if option == 0:
        return max_func(data)
    elif option == 1:
        return min_func(data)
    elif option == 2:
        return average_func(data)
    elif option == 3:
        return max_centered_func(data)
    elif option == 4:
        return diag_plus_vert_avg(data,size_x,size_y)
    elif option == 5:
        return diag_plus_vert_max_centered(data,size_x,size_y)
    elif option == 6:
        return diag_vert_max(data,size_x,size_y)
    else:
        return diag_vert_min_avg(data,size_x,size_y)

    print('FAILED!!')
    exit(2)


####################################################
#                                                  #
#                   SLIDE_WINDOW                   #
#                                                  #
####################################################

def slide_window(data,labels,col,rows,stride_x,stride_y,size_x,size_y,option):
    print("\nStarted to use a window of size ",size_x,"x",size_y," with strides ",stride_x," and ",stride_y,"!\n")
    new_data = []
    new_img_size_x = int((col-size_x)/stride_x +1)
    new_img_size_y = int((rows-size_y)/stride_y +1)
    i=1
    for image in data: # para cada imagem dos dados
        print('Images shrinked: %d \r'%(i),end='');i += 1
        new_img = np.zeros([(new_img_size_y)*(new_img_size_x)])
        for y in range(new_img_size_y):
            for x in range(new_img_size_x):
                placement_x = x*stride_x
                reconstructed_image = np.array([])
                for line in range(size_y):
                    placement_y = (y*stride_y)*36 + line * 36
                    reconstructed_image = np.concatenate((reconstructed_image,image[placement_y+placement_x:placement_y+placement_x+size_x]),axis=0)
                new_img[y*(new_img_size_x-1) + x] = window_func(reconstructed_image,option,size_x,size_y)
        new_data.append(new_img)

    # gravar com 85% para casos relevantes!!

    print('\nAll images shrinked!')
    new_data = np.array(new_data)
    return new_data, new_img_size_x,new_img_size_y

    

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




##
'''
RUN TRAIN AND TEST WITH WINDOW TECHNIQUE

(stride_x,stryde_y,window_size_x,window_size_y)

#ACCURACY TEM DE DE SER > 0.9 (NO MINIMO)

PROCURAR PARÂMETROS DA TABELA DE CONFUSÃO, CALCULAR E COMPARAR 
(falar de melhoria na previsão verificar tempos e comparar)
'''
start = time.time()

size_x = 3
size_y = 3
stride_x = 2
stride_y = 2
option = 2

new_data,new_img_size_x,new_img_size_y = slide_window(imagens,labels,n_cols,n_rows,stride_x,stride_y,size_x,size_y,option)

print(" new images size => (",new_img_size_x ,",",new_img_size_y,")")


Xt,Yt,Nt,Ne,Xe,Ye,ew,err = prep_data_train(len(new_data),new_data,labels,new_img_size_x,new_img_size_y,0.85)

'''
WINDOW 2X2
'''
if size_x == 2 and size_y == 2:
    if option == 0:    # 0.8033
        ew,err=run_stocastic(Xt,Yt,Nt,50,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
    elif option == 1:  # 0.9033
        ew,err=run_stocastic(Xt,Yt,Nt,5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 2:  # 0.8366
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.05,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.02,800,ew,err);print("\n")
    elif option == 3:  # 0.9666
        ew,err=run_stocastic(Xt,Yt,Nt,0.7,1500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,500,ew,err);print("\n")
    elif option == 4:  # 0.6766
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 5:  # 0.9633
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 6:
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    else:
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
'''
WINDOW 2X3
'''
if size_x == 2 and size_y == 3:
    if option == 0:    # 0.6533
        ew,err=run_stocastic(Xt,Yt,Nt,5,500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.03,1000,ew,err);print("\n")
    elif option == 1:  # 0.9133
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.2,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.05,500,ew,err);print("\n")
    elif option == 2:  # 0.8766
        ew,err=run_stocastic(Xt,Yt,Nt,0.9,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 3:  # 0.93
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.2,1500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,500,ew,err);print("\n")
    elif option == 4:  # 0.8833
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.05,1000,ew,err);print("\n")
    elif option == 5:  # 0.9166
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
    elif option == 6:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")
    else:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")

'''
WINDOW 3X2
'''
if size_x == 3 and size_y == 2:
    if option == 0:    # 0.7133
        ew,err=run_stocastic(Xt,Yt,Nt,10,400,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 1:  # 0.9133
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1500,ew,err);print("\n")
    elif option == 2:  # 0.82
        ew,err=run_stocastic(Xt,Yt,Nt,0.2,2000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.05,1000,ew,err);print("\n")
    elif option == 3:  # 0.9433
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,500,ew,err);print("\n")
    elif option == 4:  # 0.8733
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.2,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.05,1000,ew,err);print("\n")
    elif option == 5:  # 0.94
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
    elif option == 6:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")
    else:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")

'''
WINDOW 3X3
'''
if size_x == 3 and size_y == 3:
    if option == 0:    # 0.7366
        ew,err=run_stocastic(Xt,Yt,Nt,1,1500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1500,ew,err);print("\n")
    elif option == 1:  # 0.8733
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 2:  # 0.8566
        ew,err=run_stocastic(Xt,Yt,Nt,0.01,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.003,500,ew,err);print("\n")
    elif option == 3:  # 0.9233
        ew,err=run_stocastic(Xt,Yt,Nt,0.7,1500,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1500,ew,err);print("\n")
    elif option == 4:  # 0.87(2x2) 0.8166(1x1)
        ew,err=run_stocastic(Xt,Yt,Nt,0.7,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.3,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 5:  # 0.93(2x2) 0.9366(1x1)
        ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")
    elif option == 6:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")
    else:
        ew,err=run_stocastic(Xt,Yt,Nt,1,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.5,800,ew,err);print("\n")
        ew,err=run_stocastic(Xt,Yt,Nt,0.1,800,ew,err);print("\n")

'''
VERIFICAÇÃO DE JANELA DE TAMANHO MÁXIMO PARA PERDA TOTAL DE INFORMAÇÃO ÚTIL
'''

if size_x > 3 or size_y > 3:
    ew,err=run_stocastic(Xt,Yt,Nt,1,1000,ew,err);print("\n")
    ew,err=run_stocastic(Xt,Yt,Nt,0.5,1000,ew,err);print("\n")
    ew,err=run_stocastic(Xt,Yt,Nt,0.1,1000,ew,err);print("\n")


end = time.time()
diff = end - start
minuts = int(diff / 60) 
secs  = diff - (minuts*60)


plot_error(err)

print('in-samples error=%f ' % (cost(Xt,Yt,Nt,ew)))
C = confusion(Xt,Yt,Nt,ew)
print(C)
Acc = accuracy(C)
print("Acc: ",Acc)
print("\n")

'''
RUN TEST
'''

print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
C =confusion(Xe,Ye,Ne,ew)
print(C)
Acc = accuracy(C)
print("Acc: ",Acc)

print("\n\n00:",minuts,":",int(secs))

