import networkx as nx
import numpy as np
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, LoggingHandler,\
         SentencesDataset, InputExample
from sentence_transformers.evaluation import TripletEvaluator, \
        BinaryClassificationEvaluator, SequentialEvaluator
from datetime import datetime
import logging
import IPython
import sys

print('Torch cuda is available', torch.cuda.is_available())
logging.basicConfig(                                                            
    format="%(asctime)s - %(message)s",                                         
    datefmt="%Y-%m-%d %H:%M:%S",                                                
    level=logging.INFO,                                                         
    handlers=[LoggingHandler()],                                                
) 

def edges_train_test_split(G):

    edges = np.array([(int(u), int(v), float(G[u][v]['weight'])) for u, v in G.edges()])
    n_test = int(0.1 * len(edges))  # sample n random edges  
    print('n test edges', n_test) 
    edge_idx = np.arange(G.number_of_edges()) 
    rand_idx = np.random.choice(edge_idx, n_test, replace=False)
    test_edges = edges[rand_idx]
    train_edges = edges[np.delete(edge_idx, rand_idx)]
    print('N train edges %d, N testedges %d, N nodes %d' \
          % (len(train_edges), len(test_edges), G.number_of_nodes()))
    assert len(test_edges) + len(train_edges) == len(edges)
    return train_edges, test_edges


def get_evaluators(test_edges, G, profiles, get_binary_eval=False):
    """
    Create evaluators (TripletEvaluator and optionally BinaryClassificationEvaluator)
    that handle signed edges:
      +1 edge → nodes should be similar
      -1 edge → nodes should be dissimilar
      0 edge  → ignored
    """

    test_set_triplet = []
    test_set_binary = []
    all_nodes = np.arange(G.number_of_nodes())

    for u, v, wt in test_edges:
        if wt == 0:
            continue  # skip neutral edges

        # Random negative node
        w = np.random.choice(all_nodes, 1)[0]
        pu, pv, pw = profiles[int(u)], profiles[int(v)], profiles[int(w)]

        if wt > 0:
            # +1 edge: (anchor=pu, positive=pv, negative=pw)
            test_set_triplet.append(InputExample(texts=[pu, pv, pw]))
            if get_binary_eval:
                test_set_binary.append(InputExample(texts=[pu, pv], label=1))  # similar
        else:
            # -1 edge: dissimilar — we flip roles
            # The "true" edge (u,v) should be less similar than (u,w)
            test_set_triplet.append(InputExample(texts=[pu, pw, pv]))  # note swap of pv/pw
            if get_binary_eval:
                test_set_binary.append(InputExample(texts=[pu, pv], label=0))  # dissimilar

    # --- Create Evaluators ---
    tripletEvaluator = TripletEvaluator.from_input_examples(test_set_triplet, name='test_triplet')

    if get_binary_eval:
        binaryEvaluator = BinaryClassificationEvaluator.from_input_examples(
            test_set_binary, name='test_binary'
        )
        seq_evaluator = SequentialEvaluator(
            [tripletEvaluator, binaryEvaluator],
            main_score_function=lambda scores: np.mean(scores)
        )
        return seq_evaluator
    else:
        return tripletEvaluator


        
def rbert_signed(model_name, profiles, G):
    model = SentenceTransformer(model_name)
    print('Loading model dataset')
    train_edges, test_edges = edges_train_test_split(G)
    
    train_set = []
    for u, v, w in train_edges:
        # +1 edge → similar (label=1)
        # -1 edge → dissimilar (label=0)
        # 0 edge → neutral (ignore)
        if w > 0:
            label = 1.0
        elif w < 0:
            label = 0.0
        else:
            continue  # skip neutral edge

        train_set.append(InputExample(texts=[profiles[int(u)], profiles[int(v)]], label=label))

    train_dataset = SentencesDataset(train_set, model=model)

    # Use ContrastiveLoss for binary (similar/dissimilar) training
    train_loss = losses.ContrastiveLoss(model=model)

    
    print('prepping evaluators')
    evaluator = get_evaluators(test_edges[:, :3], G, profiles, get_binary_eval=True)
    return model, train_dataset, train_loss, evaluator


def build_rbert_model(sampling_type, base_model, profiles, G, output_path,
                      seed=2020, batch_size=32, num_epochs=2, eval_steps=10000):
    torch.manual_seed(seed)
    model_func = rbert_signed
    model, train_dataset, train_loss, evaluator = model_func(base_model, profiles, G)
    
    start_time = time.time()
    train_dataloader = DataLoader(train_dataset, 
                                  shuffle=True, 
                                  batch_size=batch_size)
   
    # 10% of train data
    warmup_steps = int(len(train_dataset) * num_epochs / batch_size * 0.1) 
    print('Number of training steps', len(train_dataloader))
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=eval_steps,
        warmup_steps=warmup_steps,
        output_path=output_path
    )
    print('Done training (%ds)' % (time.time() - start_time))
    return model

    
