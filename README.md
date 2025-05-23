# CS471 Team 12 Project
Participators: Sangheon Kang, Eunjae Jung, Seonghun Choi 

Project Type: Research 

Project link(Colab file): https://colab.research.google.com/drive/1VTIPOZXYCEL_sW_bBkO0qExSygjg_TyV?usp=sharing

## Result Reproducing Method 
* You can reproduce our experiments on free version of Google Colab environment.
* You should **mount** your drive.
* You should connect to the **T4 GPU** Runtime.
* You should place the dataset(`dgraphfin.npz`) to **appropriate path**. 
    * default: `./dataset/DGraphFin/raw/dgraphfin.npz`
    * **rule: 'current path' + '`/dataset/DGraphFin/raw/dgraphfin.npz`'**
* Fundamentally, if you execute the cells step by step, it will work. 
* You can run our models with call `main()`
    * You can transfer the arguments for model via arguments of main()
    * `main(device = 0, dataset = "DGraphFin", log_steps = 10, model = "mlp", epochs = 200, runs = 3, fold = 0, trgcn_expand_dim = 5)`
        * device: 0 is gpu, 1 is cpu 
        * dataset: the name of folder in `/dataset`
        * log_steps: the interval of logging 
        * model: model you use 
        * epochs: the number of epochs
        * runs: the number of runs (repeat the same experiments)
* Our project is motivated by [DGraphXinye Group](https://github.com/DGraphXinye/DGraphFin_baseline)
### Details
1. Mount
2. Install requirements(torch-scatter, torch-sparse, torch-geometric)
3. Change the current path 
4. Model definition (MLP, GCN, SAGE, RGCN, TGCN, TRGCN)
5. Load Utils (Evaluator, utils, logger)
6. Define Train & Test
7. Experiment 1 (applying skip connection)
8. Experiment 2 (Implementing TGCN and TRGCN)

## Information for Dataset 
(Source: https://dgraph.xinye.com/dataset)

File **dgraphfin.npz** including below keys:  

* **x**: 17-dimensional node features.
* **y**: node label.  
    There four classes. Below are the nodes counts of each class.     
    0: 1210092    
    1: 15509    
    2: 1620851    
    3: 854098    
    Nodes of Class 1 are fraud users and nodes of 0 are normal users, and they the two classes to be predicted.    
    Nodes of Class 2 and Class 3 are background users.    
    
* **edge_index**: shape (4300999, 2).   
    Each edge is in the form (id_a, id_b), where ids are the indices in x.        

* **edge_type**: 11 types of edges. 
    
* **edge_timestamp**: the desensitized timestamp of each edge.
    
* **train_mask, valid_mask, test_mask**:  
    Nodes of Class 0 and Class 1 are randomly splitted by 70/15/15.  