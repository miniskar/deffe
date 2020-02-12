from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pdb
from keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
from hierarchy_cluster import *
from base_network import *

from keras.models import Sequential
tf.keras.backend.clear_session()  # For easy reset of notebook state.

from tensorflow import keras
from tensorflow.keras import layers

class ClassificationNetwork(BaseNetwork):
    def __init__(self, args, parameters_data, cost_data, name="network"):
        self.args = args
        self.parameters_data = parameters_data
        self.cost_data = cost_data
        samples_count = int(args.n)
        classification_bins = 1000
        if samples_count != -1:
            classification_bins = min(classification_bins, samples_count)
        unique_elements = len(set(cost_data.reshape(cost_data.size).tolist()))
        if unique_elements < classification_bins:
            classification_bins = unique_elements
        self.classification_bins = classification_bins
        #self.cost_bin, self.labels = self.GetEqualDistributionBins()
        #self.cost_bin, self.labels = self.GetKMeansDistributionBins()
        self.cost_bin, self.labels = self.GetHierarchicalDistributionBins(self.classification_bins)
        print("Costbin:"+str(self.cost_bin))
        print("labels:"+str(self.labels))
        self.n_out_fmaps = classification_bins
        hidden_layers = [(64,  'relu', 'dense1'),
                         (128, 'relu', 'dense2'),
                         (256, 'relu', 'dense3'),
                         (512, 'relu', 'dense4'),
                         (self.n_out_fmaps, 'softmax', 'predict')]
        self.n_in_fmaps = parameters_data.shape[1]
        self.name = name
        self.model = BaseNetwork.get_uncompiled_model(self, self.n_in_fmaps, hidden_layers, name)
        self.model = BaseNetwork.get_compiled_model(self, self.model)
        self.batch_size = min(self.parameters_data.shape[0], int(args.batch_size))
        self.nepochs = int(args.epochs)

    def GetEpochs(self):
        return self.nepochs

    def GetBatchSize(self):
        return self.batch_size

    def basic_loss_function(self, y_true, y_pred):
        return tf.math.reduce_mean(y_true - y_pred)

    def GetEqualDistributionBins(self):
        cost_data = self.cost_data
        classification_bins = self.classification_bins
        print("Cost shape:"+str(cost_data.shape))
        min_cost = np.min(cost_data)
        max_cost = np.max(cost_data)
        cost_bin = ((cost_data-min_cost)/(max_cost-min_cost)*classification_bins).astype('int')
        return cost_bin, cost_bin

    def GetHierarchicalDistributionBins(self, classification_bins):
        cost_data = self.cost_data
        print("Cost shape:"+str(cost_data.shape))
        min_cost = np.min(cost_data)
        max_cost = np.max(cost_data)
        cost_data_rs = cost_data.reshape(cost_data.size)
        index_list = [index for index in range(len(cost_data_rs.tolist()))]
        cost_list = cost_data_rs.tolist()
        if True:
            plt.scatter(index_list, cost_list,c='black')
            plt.xlabel('Index')
            plt.ylabel('Cycles')
            plt.savefig("cycles_sorted.png")
            plt.close()
        hierarchical = Hierarchical(n_clusters = classification_bins)
        hierarchical.fit(cost_data.tolist())
        colors = mcolors.CSS4_COLORS
        colors_key = colors.keys()
        if True:
            plt.scatter(index_list, cost_list, c=hierarchical.labels_.astype(float))
            plt.savefig("cluster.png")
            plt.close()
        cost_bin = hierarchical.cluster_centers_
        hierarchical.labels_.astype(int)
        return cost_bin, hierarchical.labels_.reshape((hierarchical.labels_.size,1))


    def GetKMeansDistributionBins(self):
        cost_data = self.cost_data
        classification_bins = self.classification_bins
        print("Cost shape:"+str(cost_data.shape))
        min_cost = np.min(cost_data)
        max_cost = np.max(cost_data)
        cost_data_rs = cost_data.reshape(cost_data.size)
        index_list = [index for index in range(len(cost_data_rs.tolist()))]
        cost_list = cost_data_rs.tolist()
        if True:
            plt.scatter(index_list, cost_list,c='black')
            plt.xlabel('Index')
            plt.ylabel('Cycles')
            plt.savefig("cycles_sorted.png")
            plt.close()
        if True:
            SSE = []
            for cluster in range(1,20):
                kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
                kmeans.fit(cost_data.tolist())
                SSE.append(kmeans.inertia_)
            # converting the results into a dataframe and plotting them
            frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
            plt.figure(figsize=(12,6))
            plt.plot(frame['Cluster'], frame['SSE'], marker='o')
            plt.xlabel('Number of clusters')
            plt.ylabel('Inertia')
            plt.savefig("inertia.png")
            plt.close()
        kmeans = KMeans(n_jobs = -1, n_clusters = classification_bins, init='k-means++')
        kmeans.fit(cost_data.tolist())
        colors = mcolors.CSS4_COLORS
        colors_key = colors.keys()
        if True:
            plt.scatter(index_list, cost_list, c=kmeans.labels_.astype(float))
            plt.savefig("cluster.png")
            plt.close()
        cost_bin = kmeans.cluster_centers_
        return cost_bin, kmeans.labels_.reshape((kmeans.labels_.size,1))

    def preprocess_data(self):
        parameters = self.parameters_data
        cost_labels = self.labels
        cost_centroids = self.cost_bin
        cost_groups = [[] for i in range(self.classification_bins)]
        if True:
            n, bins, patches = plt.hist(x=cost_labels.reshape(cost_labels.size), bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('My Very Own Histogram')
            plt.text(23, 45, r'$\mu=15, b=3$')
            plt.savefig("hist.png")
            plt.close()
        max_entries = 0
        for index, label in enumerate(cost_labels):
            cost_groups[label[0]].append(index)
            max_entries = max(max_entries, len(cost_groups[label[0]]))
        train_count = int(parameters.shape[0]*80/100.0)
        test_count = parameters.shape[0] - train_count
        tr_indices = []
        test_indices = []
        for index in range(max_entries):
            for cgroup in cost_groups:
                if len(cgroup) > 0 and len(tr_indices) < train_count:
                    tr_indices.append(cgroup.pop())
                if len(cgroup) > 0 and len(test_indices) < test_count:
                    test_indices.append(cgroup.pop())
        training_idx = np.array(tr_indices).astype('int')
        test_idx = np.array(test_indices).astype('int')
        #indices = np.random.permutation(parameters.shape[0])
        print("Total count:"+str(parameters.shape[0]))
        print("Train count:"+str(train_count))
        print("Test count:"+str(test_count))
        cost_values = np.array([cost_centroids[label].tolist() for label in cost_labels]).reshape((cost_labels.shape[0], 1))
        print("Tr_indices:"+str(training_idx))
        print("test_indices:"+str(test_idx))
        x_train = parameters[training_idx,:].astype('float')
        y_train = np.zeros((x_train.shape[0], cost_centroids.shape[0]))
        z_train = cost_values[training_idx,:].astype('float')
        x_test = parameters[test_idx,:].astype('float')
        y_test = np.zeros((x_test.shape[0], cost_centroids.shape[0]))
        z_test = cost_values[test_idx,:].astype('float')

        for index, lab in enumerate(cost_labels[training_idx,:]):
            y_train[index][lab] = 1
        for index, lab in enumerate(cost_labels[test_idx,:]):
            y_test[index][lab] = 1
        x_val = x_train
        y_val = y_train
        z_val = z_train
        self.x_train, self.y_train, self.z_train = x_train, y_train, z_train
        self.x_val, self.y_val, self.z_val = x_val, y_val, z_val
        self.x_test, self.y_test, self.z_test = x_test, y_test, z_test

    def run_model(self):
        x_train, y_train, z_train = self.x_train, self.y_train, self.z_train
        x_val, y_val, z_val       = self.x_val, self.y_val, self.z_val      
        x_test, y_test, z_test    = self.x_test, self.y_test, self.z_test   
        # Train the model by slicing the data into "batches"
        # of size "batch_size", and repeatedly iterating over
        # the entire dataset for a given number of "epochs"
        if self.args.evaluate and self.args.icp != "":
            print("Loading checkpoint file:"+self.args.icp)
            self.model.load_weights(self.args.icp)
            print('\n# Evaluate on test data')
            results = self.model.evaluate(x_test, y_test, batch_size=self.GetBatchSize())
            print('test loss, test acc:', results)
        if self.args.icp != "":
            print("Loading checkpoint file:"+self.args.icp)
            self.model = keras.models.load_model(self.args.icp)
            print('\n# Evaluate on test data')
            results = self.model.evaluate(x_test, y_test, batch_size=self.GetBatchSize())
            print('test loss, test acc:', results)
        if not self.args.evaluate:
            filepath="weights-improvement-{epoch:02d}-acc{acc:.2f}-valacc{val_acc:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=False)
            callbacks_list = [checkpoint]
            print('# Fit model on training data')
            history = self.model.fit(x_train, y_train,
                                batch_size=self.GetBatchSize(),
                                epochs=self.GetEpochs(),
                                # We pass some validation for
                                # monitoring validation loss and metrics
                                # at the end of each epoch
                                validation_data=(x_test, y_test),
                                #validation_split = 0.2,
                                callbacks=callbacks_list)

            self.model.save_weights('model.hdf5')
            print(history.history.keys())

            # Plot training & validation loss values
            plot_loss = True 
            if plot_loss:
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                #plt.yscale('log')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.savefig("loss.png")
                plt.close()

            # The returned "history" object holds a record
            # of the loss values and metric values during training
            print('\nhistory dict:', history.history)

        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        results = self.model.evaluate(x_test, y_test, batch_size=self.GetBatchSize())
        print('test loss, test acc:', results)

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        print('\n# Generate predictions for 3 samples')
        predictions = self.model.predict(x_test[:3])
        print('predictions shape:', predictions.shape)

