import os
from datetime import datetime
import matplotlib.pyplot as plt
from pymongo import MongoClient
import numpy as np
import nengo
import time
import math
class Nengo_MongoDB:
    """
    Class to collect data from simulation and store it somewhere
    """
    def __init__(self):
        self.sim = None
        self.backend = ""
        self.MongoDB = MongoDB()
        self.MongoClient = None
        self.fig = None 
        self.globalTime = 0
        self.img = None
        self.simName = ""
        self.Dataset = ""
        self.Labels = None
        self.LabelsCollection = None
        self.LabelsData = []
        self.SpikesCollection = None
        self.SpikesData = []
        self.WeightsCollection = None
        self.WeightsData = []
        self.WeightsDataTMP = []
        self.weightV2 = []
        self.PotentialCollection = None 
        self.PotentialData = []
        self.PotentialDataTMP = []
        self.step_time = None
        self.input_nbr = None
        self.date = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

    # Log functions --------------------------

    def InfoAndArchi(self):
        InfoCollection = self.MongoClient[self.simName]["info"]
        InfoCollection.insert_one({"n":self.sim.model.toplevel.label,"L:N":{E.label:E.n_neurons for E in self.sim.model.toplevel.ensembles},"T":self.date,"D":self.Dataset})
        
        ArchiCollection = self.MongoClient[self.simName]["archi"]
        L = {"Layers":[{"L":E.label,"N":E.n_neurons,"NT":str(E.neuron_type),"D":str(E.dimensions)} for E in self.sim.model.toplevel.ensembles]}

        C = {"Synapses":[{"pre":C.pre.ensemble.label if(not isinstance(C.pre,nengo.node.Node)) else C.pre.label,"post":C.post.ensemble.label if(not isinstance(C.post,nengo.node.Node)) else C.post.label,"lr":str(C.learning_rule_type),"trans":str(C.transform)} for C in self.sim.model.toplevel.connections[1:]]}

        ArchiCollection.insert_many([L,C])

    # ----------------------------------------

    def set(self,sim,Args,weightVar = None):
        """
        Set the simulation argument

        Parameters
        ----------
        sim : Simulator
            Simulator argument
        Args : Collection
            Collection of data related to the simulation
        """
        
        self.sim = sim
        self.backend = Args["backend"]
        self.Dataset = Args["Dataset"]
        self.Labels = Args["Labels"]
        self.step_time = Args["step_time"]
        self.input_nbr = Args["input_nbr"]
        self.simName = self.sim.model.toplevel.label.replace(" ","-")+"-"+self.date
        self.weightV2 = weightVar

        # Read Mongo credentials 
        self.MongoClient = self.MongoDB.MongoConnect()
        assert self.MongoClient , "Can't Connect to MongoDB !"
        print("Connected to MongoDB successfully")

        if self.MongoDB.CreateDataBase(self.MongoClient,self.simName):
            print("Database created")
        else:
            print("Database already exists")

        # Store info and network arch
        self.InfoAndArchi()
        
        # Initialize Client for other Collections 
        self.LabelsCollection = self.MongoClient[self.simName]["labels"]
        self.SpikesCollection = self.MongoClient[self.simName]["spikes"]
        self.WeightsCollection = self.MongoClient[self.simName]["synapseWeight"]
        self.PotentialCollection = self.MongoClient[self.simName]["potential"]

    # Helper functions -----------------------
    
    def DrawRealTime(self,fig,img,data,t):
        """
        Draw HeatMap in realtime
        """
        # TODO: Add layer info to avoid confusion in multilayer ones 

        if(t % 41 == 0):
            print("--")
            time.sleep(2)
            img.set_array(np.random.random((50,50)))
            # redraw the figure
            fig.canvas.draw()

    def storeToMongo(self,label,data):
        # Collect for couple of steps
        # Add to Mongo
        print("Mongo processing data")
    
    def storeLabels(self,t):
        # Collect Labels for couple of steps
        if(round(round(t,3) % self.step_time,2) == 0):
            self.LabelsData.append({"L":int(str(self.Labels[int(round(t,2) / self.step_time)-1])),"T":round(round(t,2)-round(self.step_time,2),2),"G":int(round(t,2) / self.step_time)})
        
        # Add to Mongo
        if((len(self.LabelsData) == self.input_nbr * 0.25)):
            self.LabelsCollection.insert_many(self.LabelsData)
            self.LabelsData.clear()

    def storeSpikes(self,t,Probe_spikes):
        # Collect Spikes for couple of steps
        # import numpy as np
        # print(np.sum(self.sim._sim_data[Probe_spikes]))
        if(len(self.sim._sim_data[Probe_spikes])!=0):

            for i,n in enumerate(self.sim._sim_data[Probe_spikes][0].tolist()):
                if(n != 0):
                    self.SpikesData.append({"i": {"L": Probe_spikes.target.ensemble.label,"N":i},"T":round(t,3),"Input":int(str(self.Labels[int(round(t,2) / self.step_time)-1])),"S":int(n*self.sim.dt)})
        # Add to Mongo
        if(len(self.SpikesData) > 1000):
            self.SpikesCollection.insert_many(self.SpikesData)
            self.SpikesData.clear()

    def storeWeights(self,t,Probe_weights):
        # Collect Weights for couple of steps

        if(len(self.WeightsDataTMP) == 0):
            for i,f in enumerate(self.sim._sim_data[Probe_weights][0].tolist()):
                for j,v in enumerate(f):
                    x,y = self.getXY(j)
                    self.WeightsData.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": str(Probe_weights.label).split("_")[0]} )
                    self.WeightsDataTMP.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": str(Probe_weights.label).split("_")[0]} )
        else:
            for i,f in enumerate(self.sim._sim_data[Probe_weights][0].tolist()):
                for j,v in enumerate(f):
                    x,y = self.getXY(j)
                    if(self.WeightsDataTMP[i*784 + j]["V"] != round(v,9) and self.WeightsDataTMP[i*784 + j]["T"] != round(t,3)):
                        self.WeightsData.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": str(Probe_weights.label).split("_")[0]} )
                        self.WeightsDataTMP[i*784 + j] = {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": str(Probe_weights.label).split("_")[0]}
     
        # Add to Mongo
        if(len(self.WeightsData) > 10000):
            self.WeightsCollection.insert_many(self.WeightsData)
            self.WeightsData.clear()

    def storeWeightsV2(self,t,weights):
        # Collect Weights for couple of steps
        # 423 ms ± 7.91 ms per loop (mean ± std. dev. of 10 runs, 1 loop each) normal one
        # print(len(self.WeightsDataTMP))
        # print(np.sum(self.WeightsDataTMP))
        if(len(self.WeightsDataTMP) == 0):
            for i,f in enumerate(weights[0].tolist()):
                for j,v in enumerate(f):
                        x,y = self.getXY(j)
                        self.WeightsData.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": "Layer1"} )
                        self.WeightsDataTMP.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": "Layer1"})
        else:
            for i,f in enumerate(weights[0].tolist()):
                for j,v in enumerate(f):
                    x,y = self.getXY(j)
                    if(self.WeightsDataTMP[i*784 + j]["V"] != round(v,9) and self.WeightsDataTMP[i*784 + j]["T"] != round(t,3)): #
                        self.WeightsData.append( {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": "Layer1"} )
                        self.WeightsDataTMP[i*784 + j] = {"T":round(t,3), "C": j, "To":i,"V":round(v,9), "index": {"x":x ,"y":y},"L": "Layer1"}
        # Add to Mongo
        if(len(self.WeightsData) > 10000):
            self.WeightsCollection.insert_many(self.WeightsData)
            self.WeightsData.clear()

    def getXY(self,v):
        X = int(v / 28)
        Y = v % 28
        return X,Y

    def storePotential(self,t,Probe_potential):
        # Collect Potential for couple of steps
        if(len(self.sim._sim_data[Probe_potential])!=0):
            if(len(self.PotentialDataTMP) == 0):
                for i,v in enumerate(self.sim._sim_data[Probe_potential][0].tolist()):
                        self.PotentialData.append({"T":round(t,3),"L": Probe_potential.target.ensemble.label,"N":i,"V":round(v,9)})
                        self.PotentialDataTMP.append({"T":round(t,3),"L": Probe_potential.target.ensemble.label,"N":i,"V":round(v,9)})
            else:
                for i,v in enumerate(self.sim._sim_data[Probe_potential][0].tolist()):
                        if(self.PotentialDataTMP[i]["V"] != round(v,9) and self.PotentialDataTMP[i]["T"] != round(t,3)):
                            self.PotentialData.append({"T":round(t,3),"L": Probe_potential.target.ensemble.label,"N":i,"V":round(v,9)})
                            self.PotentialDataTMP[i] = {"T":round(t,3),"L": Probe_potential.target.ensemble.label,"N":i,"V":round(v,9)}
        # Add to Mongo
        if(len(self.PotentialData) > 10000):
            self.PotentialCollection.insert_many(self.PotentialData)
            self.PotentialData.clear()

    def storeTestInfo(self,Acc,Labels,outputLayer):
        InfoCollection = self.MongoClient[self.simName]["info"]
        InfoCollection.insert_one({"MaxS":Acc,"NLabel":[{"L" : outputLayer, "N" : int(K), "Label" : int(Labels[K])} for K in Labels]})
    def closeLog(self):
        """
        Close log file 
        """
        # Add any left data
        if(len(self.LabelsData)!=0):
            self.LabelsCollection.insert_many(self.LabelsData)
        if(len(self.SpikesData)!=0):
            self.SpikesCollection.insert_many(self.SpikesData)
        if(len(self.WeightsData)!=0):
            self.WeightsCollection.insert_many(self.WeightsData)
        if(len(self.PotentialData)!=0):
            self.PotentialCollection.insert_many(self.PotentialData)

        # Close Mongo client
        print("Closing Client")
        self.MongoClient.close()

    # ----------------------------------------

    # Process data step by step

    def __call__(self, t):
        
        if self.sim is not None:
            assert len(self.sim.model.probes) != 0 , "No Probes to store"


            for probe in self.sim.model.probes:
                # print(probe.attr)

                if(self.backend == "Nengo"):
                    if len(self.sim._sim_data[probe]) != 0: 
                        self.sim._sim_data[probe] = [self.sim._sim_data[probe][-1]]
                else:
                    if len(self.sim.model.params[probe]) != 0: 
                        self.sim.model.params[probe] = [self.sim.model.params[probe][-1]]
                #  Process simulation data
                
                if (probe.attr == "spikes"):
                    # print('spikes storing')
                    self.storeSpikes(t,probe)
                if (probe.attr == "voltage"):
                    self.storePotential(t,probe)
                
                if math.isclose(math.fmod(t,0.1),0,abs_tol=1e-3):
                    # print('Inside')
                    if (probe.attr == "weights"):
                        # print('Inside 1.0')
                        self.storeWeights(t,probe)

                    # print(bool(self.weightV2 != None))
                    # print(len(self.weightV2) != None)
                    # print(not isinstance(self.weightV2[0],int))
                    # print(self.weightV2[0])
                    if (self.weightV2 != None and len(self.weightV2) != None and not isinstance(self.weightV2[0],int)):
                        # print('Inside 2.0')
                        self.storeWeightsV2(t,self.weightV2)
            
            self.storeLabels(t)

# ---------------------------------------------
#   MongoDB class
# ---------------------------------------------
class MongoDB:
    USERNAME = ""
    PASSWORD = "" 
    DATABASE_URL = "mongodb://127.0.0.1:27017" 


    def MongoConnect(self):
        """ Connect to MongoDB

        Returns:
            MongoDB client instance
        """
        try:
            client = None

            if(self.USERNAME == "" and self.PASSWORD == ""):
                client = MongoClient()
                
            else:
                client = MongoClient(self.DATABASE_URL,
                username=self.USERNAME,
                password=self.PASSWORD,
                authSource='admin',
                authMechanism='SCRAM-SHA-1')
            
            return client            
            
        except Exception as e:
            print("MongoConnect:" + str(e))
            return None

    def CreateDataBase(self,client : MongoClient,name):
        """ Create a new Database

        Args:
            client (MongoClient): MongoDB Client instance
            name (String): Simulation label

        Returns:
            Execution status
        """
        dblist = client.list_database_names()
        if name not in dblist:
            client[name]
            return 1
        else:
            return 0
