import numpy as np
import time
import torch as torch
import os
from torch import nn
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, max_error, explained_variance_score
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_

device = torch.device("cuda")

class StatePredictionDataset(torch.utils.data.Dataset):
  
    def __init__(self, data_path, file_name):
        results_file = os.path.join(data_path, file_name)
        self.raw_data = np.load(results_file)
        self.state = self.raw_data['state']
        self.action = self.raw_data['action']
        self.reward = self.raw_data['reward']
        self.next_state = self.raw_data['next_state']
        self.absorbing = self.raw_data['absorbing']
        self.last = self.raw_data['last']

        self._create(self.state, self.absorbing)


    def pose_to_transformation_matrix(self, pose):
        r = Rotation.from_quat([pose[3], pose[4], pose[5], pose[6]])
        T = np.identity(4)
        T[:3,:3] = r.as_matrix()
        T[0,3] = pose[0] 
        T[1,3] = pose[1] 
        T[2,3] = pose[2]
        return T

    def transformation_matrix_to_pose(self, T):
        r = Rotation.from_matrix(T[:3,:3])
        quat = r.as_quat()
        pose = np.array([T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]])
        return pose

    def rotate_x(self, t):
        rot_x = np.eye(4)
        rot_x[1,1] = np.cos(t)
        rot_x[1,2] = -np.sin(t)
        rot_x[2,1] = np.sin(t)
        rot_x[2,2] = np.cos(t)
        return rot_x

    def rotate_y(self, t):
        rot_y = np.eye(4)
        rot_y[0,0] = np.cos(t)
        rot_y[0,2] = np.sin(t)
        rot_y[2,0] = -np.sin(t)
        rot_y[2,2] = np.cos(t)
        return rot_y

    def rotate_z(self, t):
        rot_z = np.eye(4)
        rot_z[0,0] = np.cos(t)
        rot_z[0,1] = np.sin(t)
        rot_z[1,0] = np.sin(t)
        rot_z[1,1] = np.cos(t)
        return rot_z

    def normalize(self, _d, to_sum=True, copy=True):
        # d is a (n x dimension) np array
        d = _d if not copy else np.copy(_d)
        d -= np.min(d, axis=0)
        d /= (np.sum(d, axis=0) if to_sum else np.ptp(d, axis=0))
        return d


    def _create(self, state, absorbing):
        absorbing = absorbing.astype(int)
        self.data = np.zeros((len(state), 14))
        self.data_min = np.zeros((len(state), 6))
        end_state_added = True
        data_idx = 0

        # Data min constant values
        # Translation z=-0.06038878
        # Rotation x=-90, y=0,z=variable
        for idx, s in enumerate(state):
            self.data[idx,0:7] = s[0:7]
            self.data_min[idx,0:2] = s[0:2]
            self.data_min[idx,2] = Rotation.from_quat(s[3:7]).as_euler('xyz')[2]

            if end_state_added:
                data_idx = idx
                end_state_added = False
            
            if absorbing[idx]:
                end_state_added = True
                for i in range(data_idx, idx+1):
                    self.data[i,7:14] = s[0:7]
                    self.data_min[i,3:5] = s[0:2]
                    self.data_min[i,5] = Rotation.from_quat(s[3:7]).as_euler('xyz')[2]


        # mean = np.mean(self.data[0:7],axis=1)
        # std = np.std(self.data[0:7],axis=1)

        # print("Max:", np.max(self.data[:,0:7], axis=0))
        # print("Min:", np.min(self.data[:,0:7], axis=0))

        self.data_min[:,0] = self.data_min[:,0]/6.0
        self.data_min[:,1] = self.data_min[:,1]/6.0
        self.data_min[:,2] = self.data_min[:,2]/np.pi

        self.data_min[:,3] = self.data_min[:,3]/6.0
        self.data_min[:,4] = self.data_min[:,4]/6.0
        self.data_min[:,5] = self.data_min[:,5]/np.pi

        self.data_min = torch.from_numpy(self.data_min).to(torch.device("cuda:0"))


    def visualize_data(self, idx):
        wTo = np.array([[ 1, 0,  0,  0],                      
                    [0,  1,  0,  0],                      
                    [0, 0,  1,  9.30385664e-02],                      
                    [ 0,  0,  0,  1]])

        next_absorb_idx = idx
        i = idx
        while True:
            if self.absorbing[i]:
                next_absorb_idx = i
                break
            else: 
                i = i + 1

        initial, terminal = self._getitem_full(idx)


        fig = plt.figure()
        ax = plt.axes()
        for i in range(idx, next_absorb_idx):
            initial, _ = self._getitem_full(i)

            riTo = self.pose_to_transformation_matrix(initial)
            riTo = np.matmul(self.rotate_z(np.pi), riTo)
            oTri = np.linalg.inv(riTo)
            wTri = np.matmul(wTo, oTri)     

            initial = self.transformation_matrix_to_pose(wTri)
            initial_rvec = Rotation.from_quat(initial[3:7]).as_euler('xyz')
            plt.quiver(initial[0], initial[1], np.cos(initial_rvec[2]) , np.sin(initial_rvec[2]), color='red')


        rtTo = self.pose_to_transformation_matrix(terminal)
        rtTo = np.matmul(self.rotate_z(np.pi), rtTo)
        oTrt = np.linalg.inv(rtTo)
        wTrt = np.matmul(wTo, oTrt)

        terminal = self.transformation_matrix_to_pose(wTrt)
        terminal_rvec = Rotation.from_quat(terminal[3:7]).as_euler('xyz')
        
        obj = self.transformation_matrix_to_pose(wTo)
        obj_rvec = Rotation.from_quat(obj[3:7]).as_euler('xyz')

        cc1 = plt.Circle([0,0], 0.4,fill=False, label='Table')

        plt.quiver(terminal[0], terminal[1], np.cos(terminal_rvec[2]) , np.sin(terminal_rvec[2]), color='green', label='Terminal state')
        # plt.quiver(obj[0], obj[1], np.cos(obj_rvec[2]) , np.sin(obj_rvec[2]))
        plt.quiver(obj[0], obj[1], 0 , 1, label='Object')

        ax.add_artist(cc1) 

        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim([-2.5,2.5])
        plt.ylim([-2.5,2.5])
        
        # plt.show()
        fig.savefig("/media/sdur/Data/RAL/state_prediction/data_vis/{}.png".format(idx))



    def __len__(self):
        return len(self.data_min)

    def _getitem_full(self, idx):
        return self.data[idx,0:7], self.data[idx,7:14]
  
    def __getitem__(self, idx):
        # return self.data[idx,0:7], self.data[idx,7:14]
        # print(idx,self.data_min[idx,0:2],  self.data_min[idx,3:5] )
        return self.data_min[idx,0:3], self.data_min[idx,3:6]

    def get_splits(self, n_test=0.01):
        # determine sizes
        test_size = round(n_test * len(self.data))
        train_size = len(self.data) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


def prepare_data(path, file_name):
    dataset = StatePredictionDataset(path, file_name)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=2048, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl

def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl, 0):
        # evaluate the model on the test set
        inputs, targets = inputs.float(), targets.float() # what about all double
        
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.cpu().detach().numpy()

        actual = targets.cpu().numpy()
        # actual = actual.reshape((len(actual), 3))
        # # round to class values
        # yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    # print("Pred:", predictions[0:5])
    # print("act:", actuals[0:5])
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    print("Predictions:", predictions[0:10])
    print("Actuals:", actuals[0:10])
    # calculate accuracy
    acc = explained_variance_score(actuals, predictions)
    print("Explained variance score (Best 1.0):", acc)
    return acc
 
# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat

class MLP(Module):

    def __init__(self, n_inputs, n_outputs):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_inputs, 128)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        self.hidden2 = Linear(128, 512)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        self.hidden3 = Linear(512, 4096)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        self.hidden4 = Linear(4096, 512)
        kaiming_uniform_(self.hidden4.weight, nonlinearity='relu')
        self.act4 = ReLU()
        self.hidden5 = Linear(512, 128)
        kaiming_uniform_(self.hidden5.weight, nonlinearity='relu')
        self.act5 = ReLU()
        self.hidden6 = Linear(128, n_outputs)
        # kaiming_uniform_(self.hidden6.weight, nonlinearity='relu')
        xavier_uniform_(self.hidden6.weight)
        # self.act6 = Sigmoid()


    def forward(self, X):
        X = self.hidden1(X)
        X = self.act1(X)

        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        X = self.hidden4(X)
        X = self.act4(X)

        X = self.hidden5(X)
        X = self.act5(X)

        X = self.hidden6(X)
        # X = self.act6(X)
        return X



def visualize_predictions(idxs):
    wTo = np.array([[ 1, 0,  0,  0],                      
                [0,  1,  0,  0],                      
                [0,  0,  1,  9.30385664e-02],                      
                [ 0,  0,  0,  1]])

    # Initialize the MLP
    mlp = MLP(3,3)
    mlp.load_state_dict(torch.load('/media/sdur/Data/RAL/state_prediction/model/state_predictor.pt'))
    mlp.to(torch.device("cuda:0"))

    # prepare dataset
    data_path = '/media/sdur/Data/RAL/task_1/9'
    file_name = 'results_200k.npz'

    dataset = StatePredictionDataset(data_path, file_name)

    font = {'family' : 'serif',
    'weight' : 'bold',
    'size'   : 25}

    plt.rc('font', **font)

    for i, idx in enumerate(idxs):
        # ip = torch.from_numpy(dataset[idx][0]).float()
        ip = dataset[idx][0].float()

        op = mlp.forward(ip)
        op = op.cpu().detach().numpy()
        actual = dataset[idx][1]

        
        ip = ip.cpu().detach().numpy()
        actual = actual.cpu().detach().numpy()

        print("Prediction:", op[0]*6, op[1]*6, op[2]*np.pi)
        print("Actual:", actual[0]*6, actual[1]*6, actual[2]*np.pi)

        initial = np.zeros((7,))
        initial[0] = ip[0]*6
        initial[1] = ip[1]*6
        initial[2] = -0.06038878
        # initial[3:7] = Rotation.from_euler('xyz', [-np.pi/2,0,ip[2]*np.pi]).as_quat()
        initial[3:7] = Rotation.from_euler('xyz', [0,0,ip[2]*np.pi]).as_quat()
        riTo = dataset.pose_to_transformation_matrix(initial)
        riTo = np.matmul(dataset.rotate_z(np.pi), riTo)
        oTri = np.linalg.inv(riTo)
        wTri = np.matmul(wTo, oTri)     

        initial = dataset.transformation_matrix_to_pose(wTri)
        initial_rvec = Rotation.from_quat(initial[3:7]).as_euler('xyz')

        terminal = np.zeros((7,))
        terminal[0] = actual[0]*6
        terminal[1] = actual[1]*6
        terminal[2] = -0.06038878
        # terminal[3:7] = Rotation.from_euler('xyz', [-np.pi/2,0,actual[2]*np.pi]).as_quat()
        terminal[3:7] = Rotation.from_euler('xyz', [0,0,actual[2]*np.pi]).as_quat()
        rtTo = dataset.pose_to_transformation_matrix(terminal)
        rtTo = np.matmul(dataset.rotate_z(np.pi), rtTo)
        oTrt = np.linalg.inv(rtTo)
        wTrt = np.matmul(wTo, oTrt)

        print("Terminal:", terminal)

        terminal = dataset.transformation_matrix_to_pose(wTrt)
        terminal_rvec = Rotation.from_quat(terminal[3:7]).as_euler('xyz')     
        
        prediction = np.zeros((7,))
        prediction[0] = op[0]*6
        prediction[1] = op[1]*6
        prediction[2] = -0.06038878
        # prediction[3:7] = Rotation.from_euler('xyz', [-np.pi/2,0,op[2]*np.pi]).as_quat()
        prediction[3:7] = Rotation.from_euler('xyz', [0,0,op[2]*np.pi]).as_quat()
        rpTo = dataset.pose_to_transformation_matrix(prediction)
        rpTo = np.matmul(dataset.rotate_z(np.pi), rpTo)
        oTrp = np.linalg.inv(rpTo)
        wTrp = np.matmul(wTo, oTrp)

        prediction = dataset.transformation_matrix_to_pose(wTrp)
        prediction_rvec = Rotation.from_quat(prediction[3:7]).as_euler('xyz')

        obj = dataset.transformation_matrix_to_pose(wTo)
        obj_rvec = Rotation.from_quat(obj[3:7]).as_euler('xyz')     
        
        fig = plt.figure()
        ax = plt.axes()

        plt.quiver(initial[0], initial[1], np.cos(initial_rvec[2]) , np.sin(initial_rvec[2]), color='red',label='Starting base pose')
        plt.quiver(terminal[0], terminal[1], np.cos(terminal_rvec[2]) , np.sin(terminal_rvec[2]), color='green',label='Correct base pose for grasping')
        plt.quiver(prediction[0], prediction[1], np.cos(prediction_rvec[2]) , np.sin(prediction_rvec[2]), color='blue',label='Predicted base pose for grasping')
        plt.quiver(obj[0], obj[1], 0 , 1, color='black',label='Object')

        ax.tick_params(axis='both',which='major',labelsize=12)
        cc1 = plt.Circle([0,0], 0.4,fill=False)
        ax.add_artist(cc1) 

        plt.axis('equal')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("X (in m)", fontsize=15)
        plt.ylabel("Y (in m)", fontsize=15)
        plt.xlim([-2.5,2.5])
        plt.ylim([-2.5,2.5])
        plt.legend(fontsize=15)
        
        # plt.show()
        fig.savefig("/media/sdur/Data/RAL/state_prediction/predictions_vis/{}.pdf".format(idx), dpi=600, bbox_inches='tight')

if __name__== "__main__":
    # Set fixed random number seed
    torch.manual_seed(42)

    # task = "train"
    task = "eval"
    # task = "test"
    # task = "vis_data"
    # task = "vis_pred"

    # Initialize the MLP
    mlp = MLP(3,3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mlp.to(device)

    # prepare dataset
    data_path = '/media/sdur/Data/RAL/task_1/9'
    file_name = 'results_200k.npz'

    dataset_train_dl, dataset_test_dl = prepare_data(data_path, file_name)

    if task == "train":
        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        # loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

        # Run the training loop
        for epoch in range(0, 50000):
        
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            
            # Set current loss value
            current_loss = 0.0
            
            # Iterate over the DataLoader for training data
            for i, data in enumerate(dataset_train_dl, 0):
            
                # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.float(), targets.float()
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = mlp(inputs)
                
                # Compute loss
                loss = loss_function(outputs, targets)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                
                # Print statistics
                current_loss += loss.item()
                # if i % 64 == 0:
                #     print('Loss after mini-batch %5d: %.6f' %
                #             (i + 1, current_loss))
                #     current_loss = 0.0
            print("Loss after epoch ", epoch, " ", current_loss)

            torch.save(mlp.state_dict(), '/media/sdur/Data/RAL/state_prediction/model/state_predictor.pt')

            if epoch % 50 == 0:
                torch.save(mlp.state_dict(), '/media/sdur/Data/RAL/state_prediction/model/state_predictor{}.pt'.format(epoch))

        # Process is complete.
        print('Training process has finished.')

        torch.save(mlp.state_dict(), '/media/sdur/Data/RAL/state_prediction/model/state_predictor.pt')
    elif task == "eval":
        mlp.load_state_dict(torch.load('/media/sdur/Data/RAL/state_prediction/model/state_predictor2000.pt'))
        evaluate_model(dataset_test_dl, mlp)

    elif task == "test":
        # prepare dataset
        data_path = '/media/sdur/Data/RAL/task_1/0'
        file_name = 'results200k.npz'

        idx = 350

        dataset = StatePredictionDataset(data_path, file_name)

        mlp.load_state_dict(torch.load('/media/sdur/Data/RAL/state_prediction/model/state_predictor.pt'))

        ip = torch.from_numpy(dataset[idx][0]).float()
        op = mlp.forward(ip)
        print("Prediction:", op[0]*6, op[1]*6, op[2]*np.pi)
        print("Actual:", dataset[idx][1][0]*6, dataset[idx][1][1]*6, dataset[idx][1][2]*np.pi)
    
    elif task == "vis_data":
        # prepare dataset
        data_path = '/media/sdur/Data/RAL/task_1/0'
        file_name = 'results200k.npz'

        dataset = StatePredictionDataset(data_path, file_name)
        
        dataset.visualize_data(90)
        dataset.visualize_data(178)
        dataset.visualize_data(195)
        dataset.visualize_data(210)
        dataset.visualize_data(450)
        dataset.visualize_data(550)

    elif task == "vis_pred":
        visualize_predictions([90,178,195,210,450,550])


    