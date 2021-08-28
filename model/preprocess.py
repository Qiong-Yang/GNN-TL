from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
import pickle
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
<<<<<<< HEAD
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
=======
subgraph_dict = defaultdict(lambda: len(subgraph_dict))
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
edge_dict = defaultdict(lambda: len(edge_dict))
radius=1
device = torch.device('cpu')
device2 = torch.device('cuda')

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
        
	
def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


<<<<<<< HEAD
def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
=======
def extract_subgraphs(radius, atoms, i_jbond_dict,
                         subgraph_dict, edge_dict):
    """Extract the subgraphs from a molecular graph
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
<<<<<<< HEAD
        nodes = [fingerprint_dict[a] for a in atoms]
=======
        nodes = [subgraph_dict[a] for a in atoms]
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
<<<<<<< HEAD
            The updated node IDs are the fingerprint IDs.
=======
            The updated node IDs are the subgraph IDs.
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
<<<<<<< HEAD
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])
=======
                subgraph = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(subgraph_dict[subgraph])
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def create_dataset_randomsplit(x,y,dataname):
    '''
    X=[]
    for s in range(len(x)):
        if x[s].find('+') >=0:
            pass
        elif x[s].find('-') >=0:
            pass
        elif x[s].find('a') >=0:
            pass
        elif x[s].find('i') >=0:
            pass
        elif x[s].find('B') >=0:
            pass
        elif x[s].find('Au') >=0:
            pass
        elif x[s].find('P') >=0:
            pass
        elif x[s].find('.') >=0:
            pass
        else:
            X.append(x[s])
            '''
    dataset = []  
    """Exclude the data contains '.' in its smiles."""
    x= [data for data in x
                        if '.' not in data.split()[0]]
    for i in range(len(x)):
        smiles=x[i]
<<<<<<< HEAD
        property=y[i]         
        """Create each data with the above defined functions."""
        mol = Chem.MolFromSmiles(smiles)
=======
        property=y[i]
        if '.' in smiles:
            continue
        
        mol = Chem.MolFromSmiles(smiles)		
        """Create each data with the above defined functions."""
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
        if mol is None:
            continue           
        else:
            smi = Chem.MolToSmiles(mol)            
       
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
<<<<<<< HEAD
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size, property))
    dir_dataset='/data/dict/'
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
=======
        subgraphs = extract_subgraphs(radius, atoms, i_jbond_dict,
                                                subgraph_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        subgraphs = torch.LongTensor(subgraphs).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,subgraphs, adjacency, molecular_size, property))
    dir_dataset='/data/dict/'
    dump_dictionary(subgraph_dict, dir_dataset +dataname+ '-subgraph_dict.pickle')
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    return dataset

def transferlearning_dataset(file_name):
    
<<<<<<< HEAD
    dir_input = ('/data/dict/HILIC-')
=======
    dir_input = ('/dict/')
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
<<<<<<< HEAD
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
=======
    with open(dir_input + 'subgraph_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            subgraph_dict.get(k)
            subgraph_dict[k]=c[k]
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    
    dir_dataset = '/data/'
    print(file_name)
    """Load a dataset."""
    with open(dir_dataset + file_name, 'r') as f:
        smiles_property = f.readline().strip().split()
        data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles."""
    data_original = [data for data in data_original
                        if '.' not in data.split()[0]]
    dataset = []
    for data in data_original:

        smiles, property = data.strip().split()
<<<<<<< HEAD

        """Create each data with the above defined functions."""
        mol=Chem.MolFromSmiles(smiles)
=======
        if '.' in smiles:
            continue
        
        mol = Chem.MolFromSmiles(smiles)		
        """Create each data with the above defined functions."""
        if mol is None:
            continue           
        else:
            smi = Chem.MolToSmiles(mol)  
		

        """Create each data with the above defined functions."""
        mol=Chem.MolFromSmiles(smi)
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
        smi=Chem.MolToSmiles(mol)
        mol= Chem.AddHs(Chem.MolFromSmiles(smi))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
<<<<<<< HEAD
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))            
            #Transform the above each data of numpy
            #to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size, property))
    dir_dataset='/data/dict/HILIC-tf-'
    dump_dictionary(fingerprint_dict, dir_dataset + '-fingerprint_dict.pickle')
=======
        subgraphs = extract_subgraphs(radius, atoms, i_jbond_dict,
                                                subgraph_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))            
            #Transform the above each data of numpy
            #to pytorch tensor on a device (i.e., CPU or GPU).
        subgraphs = torch.LongTensor(subgraphs).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,subgraphs , adjacency, molecular_size, property))
    dir_dataset='/data/dict/HILIC-tf-'
    dump_dictionary(subgraph_dict, dir_dataset + '-subgraph_dict.pickle')
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    dump_dictionary(atom_dict, dir_dataset + '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset + '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset+ '-edge_dict.pickle')
    return dataset

def transferlearning_dataset_predict(x):
    dir_input =  ('/data/dict/')
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
<<<<<<< HEAD
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
=======
    with open(dir_input + 'subgraph_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            subgraph_dict.get(k)
            subgraph_dict[k]=c[k]
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    dataset = []
    a=[]
    X=[]
    for i in range(len(x)):

        smiles=x[i]
<<<<<<< HEAD
        """Create each data with the above defined functions."""       
        mol = Chem.MolFromSmiles(smiles)
=======
        if '.' in smiles:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        """Create each data with the above defined functions."""       
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
        if mol is None:
            a.append(i)
                #print(i,inc)
            continue
            
        else:
            smi = Chem.MolToSmiles(mol)
            
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
<<<<<<< HEAD
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        #property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size))
=======
        subgraphs = extract_subgraphs(radius, atoms, i_jbond_dict,
                                                subgraph_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        subgraphs = torch.LongTensor(subgraphs).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        #property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,subgraphs, adjacency, molecular_size))
>>>>>>> 6223ed9d9eb8ba67a0836fe2803ae89b19453aea
    
    return dataset,a
    