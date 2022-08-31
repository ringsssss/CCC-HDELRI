import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from itertools import chain

from pai4sk import BoostingMachine


def getData_case(seed,path):
    ligand_dict = {}
    receptor_dict  = {}
    name_L = []
    name_R = []
    int_name = []
    test_name = []
    feature = []

    df = pd.read_csv(path + 'interaction.csv',header=0,index_col=0)
    
    #Read ligand sequences in the interaction matrix
    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])
    #Read the feature extracted from Pse-in-One
    ligand_Kmer = pd.read_table(path + 'ligand_kmer.txt',header=None).to_numpy()
    ligand_DP = pd.read_table(path + 'ligand_DP.txt',header=None).to_numpy()
    ligand_ACC = pd.read_table(path + 'ligand_ACC.txt',header=None).to_numpy()
    ligand_PC = pd.read_table(path + 'ligand_PC-PseAAC-General.txt',header=None).to_numpy()
    ligand_SC = pd.read_table(path + 'ligand_SC-PseAAC-General.txt',header=None).to_numpy()
    ligand_DT = pd.read_table(path + 'ligand_DT.txt',header=None).to_numpy()

    #Obtain ligand feature
    ligand_feature = np.hstack((ligand_Kmer,ligand_DP,ligand_ACC,ligand_PC,ligand_SC,ligand_DT))


    for i,name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]

    #Read receptor sequences in the interaction matrix
    with open(path + 'receptor.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_R.append(line[1:-1])

    receptor_Kmer = pd.read_table(path + 'receptor_kmer.txt',header=None).to_numpy()
    receptor_DP = pd.read_table(path + 'receptor_DP.txt',header=None).to_numpy()
    receptor_ACC = pd.read_table(path + 'receptor_ACC.txt',header=None).to_numpy()
    receptor_PC = pd.read_table(path + 'receptor_PC-PseAAC-General.txt',header=None).to_numpy()
    receptor_SC = pd.read_table(path + 'receptor_SC-PseAAC-General.txt',header=None).to_numpy()
    receptor_DT = pd.read_table(path + 'receptor_DT.txt',header=None).to_numpy()
    
    #Obtain receptor feature
    receptor_feature = np.hstack((receptor_Kmer,receptor_DP,receptor_ACC,receptor_PC,receptor_SC,receptor_DT))


    for i,name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    
    count1 = 0   #Number of positive samples in the training set
    count0 = 0   #Number of negative samples in the training set
    count_test = 0  #Number of samples in the testing set
    [row,column] = np.where(df.values==1) 
    for i,j in zip(row,column):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        int_name.append(df.index[i]+'-'+df.columns[j])
        feature.append(temp_f)   #Obtain feature of positive samples
        count1 += 1

    row = []
    col = []
    row_t = []
    col_t = []
    [row0,column0] = np.where(df.values==0)   
    rand = np.random.RandomState(seed)
    num = rand.randint(row0.shape,size=count1)        
    for i in num:
        row.append(row0[i])
        col.append(column0[i])
    

    for i,j in zip(row,col):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        int_name.append(df.index[i]+'-'+df.columns[j])
        feature.append(temp_f)   #Obtain feature of negative samples
        count0 += 1

    for i in range(row0.shape[0]):
        row_t.append(row0[i])
        col_t.append(column0[i])

    for i,j in zip(row_t,col_t):       
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        test_name.append(df.index[i]+' '+df.columns[j])
        feature.append(temp_f)    #Obtain feature of testing samples
        count_test += 1

    pca=PCA(n_components=0.99)   #PCA dimension reduction
    pca.fit(feature)
    feature = pca.fit_transform(feature)


    train = []
    test = []
    for i in range(count1+count0+count_test):
        if i < count1:
            train.append(list(feature[i])+[1])
        elif i< count1 + count0:
            train.append(list(feature[i])+[0])
        else:
            test.append(list(feature[i])+[0])


    return train,test,test_name


def case(seed,path):
    train,test,test_name = getData_case(seed,path)

    train = np.array(train)
    test = np.array(test)
    d = train.shape[1]  #feature dimension

    X_train = train[:,:-1]
    X_test = test[:,:-1]
    y_train = train[:,-1]

    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = Sequential()

    model.add(LSTM(d, return_sequences=True))
    model.add(LSTM(d))
    model.add(Dense(256, activation='elu')) # Full connection layer
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    model.fit(X_train_lstm, y_train,
            epochs=200,
            batch_size=128,verbose=0)
    prob_lstm = model.predict(X_test_lstm)
    prob_lstm = np.array(list(chain.from_iterable(prob_lstm)))

    model1 = BoostingMachine(objective='logloss',num_round=4000,min_max_depth=1,max_max_depth=24)  #Heterogeneous Newton Boosting Machine
    model1.fit(X_train, y_train)

    pred = []
    prob_snap = model1.predict_proba(X_test)[:,1]


    pred = []
    prob = prob_lstm*0.6 + prob_snap*0.4  #Ensemble Learning
    for k in prob:
        if k > 0.8:
            pred.append(1)
        else:
            pred.append(0)
    pred = np.array(pred)

    test_name = np.array(test_name)



    path  = './dataset 1/'
    df1 = pd.read_csv(path + 'LRI-known.csv',header=None,index_col=None).to_numpy()  # Known LRIs in the dataset,and the first column is ligand ID,the second column is receptor ID
    df2 = pd.read_csv(path + 'LRI-pred.csv',header=None,index_col=None).to_numpy()   #LRIs predicted by LRI-HDEnHD
    ll = pd.read_csv(path + 'ligand_gen.csv',header=None,index_col=None).to_numpy()  #Correspondence table of protein ID and gene
    rl = pd.read_csv(path + 'receptor_gen.csv',header=None,index_col=None).to_numpy()

    df = np.vstack((df1,df2))
    ligand = df[:,0]
    receptor = df[:,1]
    l_dict={}
    r_dict={}
    for i in range(ll.shape[0]):
        l_dict[ll[i,2]]=ll[i,1]
    for i in range(rl.shape[0]):
        r_dict[rl[i,2]]=rl[i,1]

    l_gene = []
    r_gene = []
    for i in ligand:
        l_gene.append(l_dict[i])
    for i in receptor:
        r_gene.append(r_dict[i])

    LRI_gene = np.vstack((np.array(l_gene).T,np.array(r_gene).T)).T  #LRIs used to calculate CCC score



    dt = pd.read_csv('./dataset 1/GSE72056.csv',index_col = 0,header=None)
    dict = {}
    dict["Cell"] = np.array(dt.loc["Cell"])
    for i in range(1,dt.shape[0]):
        dict[dt.index[i]] = np.array(dt.loc[dt.index[i]],dtype = float)


    #0=melanoma cancer cells,1=T cells,2=B cells,3=macrophages,4=endothelial cells,5=CAFs,6=NK cells
    malignant_index = np.where(dict['malignant'] == 2)[0]
    T_index = np.where(dict['non-malignant cell type'] == 1)[0]
    B_index = np.where(dict['non-malignant cell type'] == 2)[0]
    Macro_index = np.where(dict['non-malignant cell type'] == 3)[0]
    Endo_index = np.where(dict['non-malignant cell type'] == 4)[0]
    CAF_index = np.where(dict['non-malignant cell type'] == 5)[0]
    NK_index = np.where(dict['non-malignant cell type'] == 6)[0]


    for i in range(7):
        for j in range(7):
            exec('mult_score{}{} = 0'.format(i,j))  #The intercellular communication score based on the filtered LRIs and the expression product approach.

    for i in range(7):
        for j in range(7):
            exec('thrd_score{}{} = 0'.format(i,j)) #The intercellular communication score based on the filtered LRIs and the expression thresholding approach.

    for i in range(7):
        for j in range(7):
            exec('mult_list{}{} = []'.format(i,j))  #LRIs between two cell types in expression product approach

    for i in range(7):
        for j in range(7):
            exec('mult_list_s{}{} = []'.format(i,j))  #LRI score between two cell types in expression product approach

    for i in range(7):
        for j in range(7):
            exec('thrd_list{}{} = []'.format(i,j))  #expression thresholding approach

    for i in range(7):
        for j in range(7):
            exec('thrd_list_s{}{} = []'.format(i,j))



    for i in LRI_gene:
        if i[0] in dict and i[1] in dict:
            malignant_l=1/malignant_index.shape[0]*sum(dict[i[0]][malignant_index])  #expression product approach
            malignant_r=1/malignant_index.shape[0]*sum(dict[i[1]][malignant_index])
            T_l=1/T_index.shape[0]*sum(dict[i[0]][T_index])
            T_r=1/T_index.shape[0]*sum(dict[i[1]][T_index])
            B_l=1/B_index.shape[0]*sum(dict[i[0]][B_index])
            B_r=1/B_index.shape[0]*sum(dict[i[1]][B_index])
            Macro_l=1/Macro_index.shape[0]*sum(dict[i[0]][Macro_index])
            Macro_r=1/Macro_index.shape[0]*sum(dict[i[1]][Macro_index])
            Endo_l=1/Endo_index.shape[0]*sum(dict[i[0]][Endo_index])
            Endo_r=1/Endo_index.shape[0]*sum(dict[i[1]][Endo_index])
            CAF_l=1/CAF_index.shape[0]*sum(dict[i[0]][CAF_index])
            CAF_r=1/CAF_index.shape[0]*sum(dict[i[1]][CAF_index])
            NK_l=1/NK_index.shape[0]*sum(dict[i[0]][NK_index])
            NK_r=1/NK_index.shape[0]*sum(dict[i[1]][NK_index])
            l_list = [malignant_l,T_l,B_l,Macro_l,Endo_l,CAF_l,NK_l]
            r_list = [malignant_r,T_r,B_r,Macro_r,Endo_r,CAF_r,NK_r]
            a = b = 0
            for item in product(l_list, r_list):
                exec('mult_score{}{} += {}'.format(a,b, (item[0]*item[1])))
                exec('mult_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
                exec('mult_list_s{}{}.append({})'.format(a,b,(item[0]*item[1])))
                b += 1
                if b == 7:
                    b = 0
                    a += 1

            mean_l_malignant = np.mean(dict[i[0]][malignant_index])   #expression thresholding approach
            mean_l_T = np.mean(dict[i[0]][T_index])
            mean_l_B = np.mean(dict[i[0]][B_index])
            mean_l_Macro = np.mean(dict[i[0]][Macro_index])
            mean_l_Endo = np.mean(dict[i[0]][Endo_index])
            mean_l_CAF = np.mean(dict[i[0]][CAF_index])
            mean_l_NK = np.mean(dict[i[0]][NK_index])
            mean_l = np.mean((mean_l_malignant,mean_l_T,mean_l_B,mean_l_Macro,mean_l_Endo,mean_l_CAF,mean_l_NK))
            std_l = np.std(dict[i[0]][np.concatenate((malignant_index,T_index,B_index,Macro_index,Endo_index,CAF_index,NK_index))])

            mean_r_malignant = np.mean(dict[i[1]][malignant_index])
            mean_r_T = np.mean(dict[i[1]][T_index])
            mean_r_B = np.mean(dict[i[1]][B_index])
            mean_r_Macro = np.mean(dict[i[1]][Macro_index])
            mean_r_Endo = np.mean(dict[i[1]][Endo_index])
            mean_r_CAF = np.mean(dict[i[1]][CAF_index])
            mean_r_NK = np.mean(dict[i[1]][NK_index])
            mean_r = np.mean((mean_r_malignant,mean_r_T,mean_r_B,mean_r_Macro,mean_r_Endo,mean_r_CAF,mean_r_NK))
            std_r = np.std(dict[i[1]][np.concatenate((malignant_index,T_index,B_index,Macro_index,Endo_index,CAF_index,NK_index))])

            malignant_l=int(mean_l_malignant>mean_l+std_l)
            malignant_r=int(mean_r_malignant>mean_r+std_r)
            T_l=int(mean_l_T>mean_l+std_l)
            T_r=int(mean_r_T>mean_r+std_r)
            B_l=int(mean_l_B>mean_l+std_l)
            B_r=int(mean_r_B>mean_r+std_r)
            Macro_l=int(mean_l_Macro>mean_l+std_l)
            Macro_r=int(mean_r_Macro>mean_r+std_r)
            Endo_l=int(mean_l_Endo>mean_l+std_l)
            Endo_r=int(mean_r_Endo>mean_r+std_r)
            CAF_l=int(mean_l_CAF>mean_l+std_l)
            CAF_r=int(mean_r_CAF>mean_r+std_r)
            NK_l=int(mean_l_NK>mean_l+std_l)
            NK_r=int(mean_r_NK>mean_r+std_r)
            l_list = [malignant_l,T_l,B_l,Macro_l,Endo_l,CAF_l,NK_l]
            r_list = [malignant_r,T_r,B_r,Macro_r,Endo_r,CAF_r,NK_r]
            a = b = 0
            for item in product(l_list, r_list):
                exec('thrd_score{}{} += {}'.format(a,b, int(item[0]&item[1])))
                exec('thrd_list{}{}.append("{}" + "-" + "{}")'.format(a,b,i[0],i[1]))
                exec('thrd_list_s{}{}.append({})'.format(a,b,int(item[0]&item[1])))
                b += 1
                if b == 7:
                    b = 0
                    a += 1
            
            


    for i in range(7):
        exec('mult_score{}{} = mult_score{}{}/2'.format(i,i,i,i))

    for i in range(7):
        exec('thrd_score{}{} = thrd_score{}{}/2'.format(i,i,i,i))
   

if __name__ == "__main__":
    case(10,'./dataset 1/')

