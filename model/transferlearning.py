
import sys
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.metrics import roc_auc_score
import preprocess as pp
import pickle
import pandas as pd
import numpy as np

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        Smiles,subgraphs, adjacencies, molecular_sizes = inputs
        subgraphs=[t.cuda() for t in subgraphs]
        adjacencies=[t.cuda() for t in adjacencies]
      
        subgraphs = torch.cat(subgraphs)
        adjacencies = self.pad( adjacencies, 0)
        """GNN layer (update the subgraph vectors)."""
        subgraph_vectors = self.embed_fingerprint(subgraphs)
        for l in range(layer_hidden):
            hs = self.update(adjacencies, subgraph_vectors, l)
            subgraph_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the subgraph vectors."""
        molecular_vectors = self.sum(subgraph_vectors, molecular_sizes)
        return Smiles,molecular_vectors

    def mlp(self, vectors):
        """ regressor based on multilayer perceptron."""
        for l in range(layer_output):
            vectors = torch.relu(self.W_output[l](vectors))
        outputs = self.W_property(vectors)
        return outputs
    def forward_regressor(self, data_batch, train):

        inputs = data_batch[:-1]
        a=data_batch[-1]
        a=[t.cuda() for t in a]
        correct_values = torch.cat(a)

        if train:
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            a=nn.L1Loss()
            loss = a(correct_values, predicted_values)
            return loss
        else:
            with torch.no_grad():
                Smiles,molecular_vectors = self.gnn(inputs)
                predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            correct_values = correct_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            correct_values = np.concatenate(correct_values)
            return Smiles,predicted_values, correct_values
    def forward_predict(self, data_batch):

            inputs = data_batch
            Smiles,molecular_vectors = self.gnn(inputs)
            predicted_values = self.mlp(molecular_vectors)
            predicted_values = predicted_values.to('cpu').data.numpy()
            predicted_values = np.concatenate(predicted_values)
            
            return Smiles,predicted_values
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total
class Trainer_tf(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model.forward_regressor(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model
    def test_regressor(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values,correct_values) = self.model.forward_regressor(
                                               data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_values)
            Ys.append(predicted_values)
            
            SAE += sum(np.abs(predicted_values-correct_values))
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        MAEs = SAE / N  # mean absolute error.
        return MAEs,predictions
    def test_predict(self, dataset):
        N = len(dataset)
        SMILES, Ts, Ys = '', [], []
        SAE = 0  # sum absolute error.
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            (Smiles,  predicted_values) = self.model.forward_predict(
                                               data_batch)
            SMILES += ' '.join(Smiles) + ' '
            Ys.append(predicted_values)
        SMILES = SMILES.strip().split()
        Y = map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, Y)])
        return predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')
    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def split_dataset(dataset, ratio):
#   """Shuffle and split a dataset."""
    np.random.seed(1)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]
def dump_dictionary(dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)
if __name__ == "__main__": 
    
    radius=1
    dim=48
    layer_hidden=10
    layer_output=10
    batch_train=8
    batch_test=8
    lr=2e-7
    lr_decay=0.99
    decay_interval=100
    iteration_tf=500
    N=5000
    path='/data/'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('The code uses a CPU!')
    import datetime
    time1=str(datetime.datetime.now())[0:13]
    dataset= pp.transferlearning_dataset('HILIC-train.txt')
    dataset_train, dataset_dev = split_dataset(dataset, 0.7)
    dataset_test = pp.transferlearning_dataset('HILIC-test.txt') 
    dataset_val= pp.transferlearning_dataset('HILIC-val.txt') 
    
    print('-'*100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of training data samples:', len(dataset_dev))
    print('# of development data samples:', len(dataset_val))
    print('# of test data samples:', len(dataset_test))
    #print('# of val data samples:', len(dataset_val))
    print('-'*100)
    print('Creating a model.')
    torch.manual_seed(1234)
    model= MolecularGraphNeuralNetwork(
        N, dim, layer_hidden, layer_output).to(device)
    file_model='/model/pre_GNN_model.h5'
    model.load_state_dict(torch.load(file_model))
    for para in model.W_fingerprint.parameters():
        para.requires_grad = False
    print(model)
    trainer = Trainer_tf(model)
    tester = Tester(model)
    print('# of model parameters:',
              sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)
      
    file_MAEs = path+'output_tf/'+time1+'_MAEs'+'.txt'
    file_test_result  = path+'output_tf/'+time1+ '_test_prediction'+ '.txt'
    file_val_result  = path+'output_tf/'+time1+ '_val_prediction'+ '.txt'
    file_train_result  = path+'output_tf/'+time1+ '_train_prediction'+ '.txt'
    file_model = path+ 'output_tf/'+time1+'_model'+'.h5'
    file1=path+'output_tf/'+time1+'-MAE.png'
    file2=path+'output_tf/'+time1+'pc-train.png'
    file3=path+'output_tf/'+time1+'pc-test.png'
    file4=path+'output_tf/'+time1+'pc-val.png'
       
    result_tf = 'Epoch\tTime(sec)\tLoss_train\tMAE_train\tMAE_dev\tMAE_test'
    with open(file_MAEs, 'w') as f:
        f.write(result_tf + '\n')
    print('Start training.')
    print('The result is saved in the output directory every epoch!')
    start = timeit.default_timer()
    for epoch in range(iteration_tf):
        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        model.train()
        loss_train = trainer.train(dataset_train)
        MAE_tf_best=9999999
        model.eval()
        MAE_tf_train,predictions_train_tf = tester.test_regressor(dataset_train)
        MAE_tf_dev = tester.test_regressor(dataset_dev)[0]
        MAE_tf_test = tester.test_regressor(dataset_test)[0]
        time = timeit.default_timer() - start
        if epoch == 1:
            minutes = time * iteration_tf / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                   hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(result_tf)
        results_tf = '\t'.join(map(str, [epoch, time, loss_train,MAE_tf_train,
                                     MAE_tf_dev, MAE_tf_test]))
        tester.save_MAEs(results_tf, file_MAEs)
        if MAE_tf_dev <= MAE_tf_best:
            MAE_tf_best = MAE_tf_dev
            tester.save_model(model, file_model)
        print(results_tf)
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import median_absolute_error,r2_score, mean_absolute_error,mean_squared_error
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))
    loss = pd.read_table(file_MAEs)
    plt.plot(loss['MAE_train'], color='r',label='MSE of train set')
    plt.plot(loss['MAE_dev'], color='b',label='MSE of validation set')
    plt.plot(loss['MAE_test'], color='y',label='MSE of test set')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(file1,dpi=300)
    plt.show()
    
    predictions_train = tester.test_regressor(dataset_train)[1]
    tester.save_predictions(predictions_train, file_train_result )
    predictions_test = tester.test_regressor(dataset_test)[1]
    tester.save_predictions(predictions_test, file_test_result)
    predictions_val = tester.test_regressor(dataset_val)[1]
    tester.save_predictions(predictions_val, file_val_result)
    res_tf = pd.read_table(file_train_result)
    r2 = r2_score(res_tf['Correct'], res_tf['Predict'])
    mae = mean_absolute_error(res_tf['Correct'], res_tf['Predict'])
    medae = median_absolute_error(res_tf['Correct'], res_tf['Predict'])
    rmae = np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct']) * 100
    median_re = np.median(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    mean_re=np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    plt.plot(res_tf['Correct'], res_tf['Predict'], '.', color = 'blue')
    plt.plot([0,1400], [0,1400], color ='red')
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')        
    plt.text(0,1400, 'R2='+str(round(r2,4)), fontsize=12)
    plt.text(500,1400,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(0, 1200, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(500, 1200, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(0, 1000, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.savefig(file2,dpi=300)
    plt.show()
    res_tf = pd.read_table(file_test_result)
    r2 = r2_score(res_tf['Correct'], res_tf['Predict'])
    mae = mean_absolute_error(res_tf['Correct'], res_tf['Predict'])
    medae = median_absolute_error(res_tf['Correct'], res_tf['Predict'])
    rmae = np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct']) * 100
    median_re = np.median(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    mean_re=np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    plt.plot(res_tf['Correct'], res_tf['Predict'], '.', color = 'blue')
    plt.plot([0,1400], [0,1400], color ='red')
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')        
    plt.text(0,1400, 'R2='+str(round(r2,4)), fontsize=12)
    plt.text(500,1400,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(0, 1200, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(500, 1200, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(0, 1000, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.savefig(file3,dpi=300)
    plt.show()
    
    res_tf = pd.read_table(file_val_result)
    r2 = r2_score(res_tf['Correct'], res_tf['Predict'])
    mae = mean_absolute_error(res_tf['Correct'], res_tf['Predict'])
    medae = median_absolute_error(res_tf['Correct'], res_tf['Predict'])
    rmae = np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct']) * 100
    median_re = np.median(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    mean_re=np.mean(np.abs(res_tf['Correct'] - res_tf['Predict']) / res_tf['Correct'])
    plt.plot(res_tf['Correct'], res_tf['Predict'], '.', color = 'blue')
    plt.plot([0,1400], [0,1400], color ='red')
    plt.ylabel('Predicted RT')
    plt.xlabel('Experimental RT')        
    plt.text(0,1400, 'R2='+str(round(r2,4)), fontsize=12)
    plt.text(500,1400,'MAE='+str(round(mae,4)),fontsize=12)
    plt.text(0, 1200, 'MedAE='+str(round(medae,4)), fontsize=12)
    plt.text(500, 1200, 'MRE='+str(round(mean_re,4)), fontsize=12)
    plt.text(0, 1000, 'MedRE='+str(round(median_re,4)), fontsize=12)
    plt.savefig(file4,dpi=300)
    plt.show()
   
  

    
      
    
