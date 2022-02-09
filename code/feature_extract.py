import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def getData(seed,path):
    ligand_dict = {}
    receptor_dict  = {}
    name_L = []
    name_R = []
    int_name = []
    feature = []

    df = pd.read_csv(path + 'interaction.csv',header=0,index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])

    ligand_Kmer = pd.read_table(path + 'ligand_kmer.txt',header=None).to_numpy()
    ligand_DP = pd.read_table(path + 'ligand_DP.txt',header=None).to_numpy()
    ligand_ACC = pd.read_table(path + 'ligand_ACC.txt',header=None).to_numpy()
    ligand_PC = pd.read_table(path + 'ligand_PC-PseAAC-General.txt',header=None).to_numpy()
    ligand_SC = pd.read_table(path + 'ligand_SC-PseAAC-General.txt',header=None).to_numpy()
    ligand_DT = pd.read_table(path + 'ligand_DT.txt',header=None).to_numpy()

    ligand_feature = np.hstack((ligand_Kmer,ligand_DP,ligand_ACC,ligand_PC,ligand_SC,ligand_DT))


    for i,name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]
    


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

    receptor_feature = np.hstack((receptor_Kmer,receptor_DP,receptor_ACC,receptor_PC,receptor_SC,receptor_DT))

    for i,name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    [row,column] = np.where(df.values==1)
    count1 = 0
    count0 = 0
    for i,j in zip(row,column):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        int_name.append(df.index[i]+'-'+df.columns[j])
        feature.append(temp_f)
        count1 += 1

    row = []
    col = []
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
        feature.append(temp_f)
        count0 += 1

    pca=PCA(n_components=0.99)
    pca.fit(feature)
    feature = pca.fit_transform(feature)

    print("feature_shape:",end='')
    print(np.array(feature).shape)

    train = []
    for i in range(count1+count0):
        if i < count1:
            train.append(list(feature[i])+[1])
        else:
            train.append(list(feature[i])+[0])


    return train



def getData_case(seed,path):
    ligand_dict = {}
    receptor_dict  = {}
    name_L = []
    name_R = []
    int_name = []
    test_name = []
    feature = []
    test_feature = []

    df = pd.read_csv(path + 'interaction.csv',header=0,index_col=0)

    with open(path + 'ligand.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name_L.append(line[1:-1])

    ligand_Kmer = pd.read_table(path + 'ligand_kmer.txt',header=None).to_numpy()
    ligand_DP = pd.read_table(path + 'ligand_DP.txt',header=None).to_numpy()
    ligand_ACC = pd.read_table(path + 'ligand_ACC.txt',header=None).to_numpy()
    ligand_PC = pd.read_table(path + 'ligand_PC-PseAAC-General.txt',header=None).to_numpy()
    ligand_SC = pd.read_table(path + 'ligand_SC-PseAAC-General.txt',header=None).to_numpy()
    ligand_DT = pd.read_table(path + 'ligand_DT.txt',header=None).to_numpy()

    ligand_feature = np.hstack((ligand_Kmer,ligand_DP,ligand_ACC,ligand_PC,ligand_SC,ligand_DT))



    for i,name in enumerate(name_L):
        ligand_dict[name] = ligand_feature[i]
    


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

    receptor_feature = np.hstack((receptor_Kmer,receptor_DP,receptor_ACC,receptor_PC,receptor_SC,receptor_DT))

    for i,name in enumerate(name_R):
        receptor_dict[name] = receptor_feature[i]

    [row,column] = np.where(df.values==1)
    count1 = 0
    count0 = 0
    count_test = 0
    for i,j in zip(row,column):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        int_name.append(df.index[i]+'-'+df.columns[j])
        feature.append(temp_f)
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
        feature.append(temp_f)
        count0 += 1

    for i in range(row0.shape[0]):
        row_t.append(row0[i])
        col_t.append(column0[i])

    for i,j in zip(row_t,col_t):
        lig_tri_fea = ligand_dict[df.index[i]]
        rec_tri_fea = receptor_dict[df.columns[j]]
        temp_f = list(lig_tri_fea) + list(rec_tri_fea)
        test_name.append(df.index[i]+' '+df.columns[j])
        feature.append(temp_f)
        count_test += 1
    print("count_test:%d"%count_test)


    pca=PCA(n_components=0.99)
    pca.fit(feature)
    feature = pca.fit_transform(feature)

    print("feature_shape:",end='')
    print(np.array(feature).shape)

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

if __name__ == '__main__':
    getData(1,'./human/')


