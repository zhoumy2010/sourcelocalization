import numpy as np
import networkx as nx
import random
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import torch
import pickle


def load_dataset(dataset, data_dir):
    """
    Load a dataset from a pickle file.

    Args:

    - dataset (str): The name of the dataset file, 'karate', 'dolphins', 'jazz', 'netscience', 'cora_ml', 'power_grid'.

    - data_dir (str): The directory where the dataset files are stored.

    Returns:

    - graph (dict): A dictionary containing the dataset.

    """

    data_dir = data_dir + "/data/" + dataset
    with open(data_dir, 'rb') as f:
        graph = pickle.load(f)
    return graph


def generate_seed_vector(top_nodes, seed_num, G, random_seed):
    """
    Generate a seed vector for diffusion simulation.

    Args:

    - top_nodes (list): List of top nodes based on node degree.

    - seed_num (int): Number of seed nodes.

    - G (networkx.Graph): The graph object.

    - random_seed (int): Random Seed

    Returns:

        seed_vector (list): Seed vector for diffusion simulation.
    """
    random.seed(random_seed)
    seed_nodes = random.sample(top_nodes, seed_num)
    seed_vector = [1 if node in seed_nodes else 0 for node in G.nodes()]
    return seed_vector


def diffusion_generation(
        graph,
        sim_num=10,
        diff_type='IC',
        time_step=10,
        repeat_step=10,
        seed_ratio=0.1,
        infect_prob=0.1,
        recover_prob=0,
        threshold=0.5,
        random_seed=0):
    """
    Generate diffusion matrices for a graph.

    Args:

    - graph (dict): Dictionary containing the graph information.

    - sim_num (int): Number of simulations.

    - diff_type (str): Type of diffusion model (IC, LT, SI, SIS, SIR). IC stands for Independent Cascade, LT stands for Linear Threshold, SI stands for Susceptible or Infective, SIS stands for Susceptible or Infective or Susceptible, SIR stands for Susceptible or Infective or Recovered.

    - time_step (int): Number of time steps in the simulation.

    - repeat_step (int): Number of repetitions for each simulation.

    - seed_ratio (float): Ratio of seed nodes, should be between 0 and 0.3.

    - infect_prob (float): Infection probability, used in SIS, SIR or SI.

    - recover_prob (float): Recovery probability, used in SIS or SIR.

    - threshold (float): Threshold parameter for diffusion models, used in IC or LT.

    - random_seed (int): Random seed.

    Returns:

    - dataset (dict): Dictionary containing ('adj_mat') adjacency matrix (the dimensionality is number of nodes * number of nodes) and ('diff_mat') diffusion matrices (the dimensionality is number of simulations * number of nodes * number of time steps).

    Example:

    import os

    curr_dir = os.getcwd()

    from data.utils import load_dataset, diffusion_generation

    data_name = 'karate'

    graph = load_dataset(data_name, data_dir=curr_dir)

    dataset = diffusion_generation(graph=graph, infect_prob=0.3, diff_type='IC', sim_num=100, seed_ratio=0.1)
    """
    adj_mat = graph['adj_mat']
    G = nx.from_scipy_sparse_array(adj_mat)
    node_num = len(G.nodes())
    seed_num = int(seed_ratio * node_num)
    simulation = np.zeros((sim_num, node_num, time_step))

    assert seed_ratio <= 0.3 and seed_ratio >= 0, "seed_ratio should be between 0 and 0.3"

    for i in range(sim_num):
        np.random.seed(random_seed + i)
        seed_vector = np.random.choice(G.nodes(), seed_num, replace=False)
        all_iterations = []
        config = mc.Configuration()

        if diff_type == 'LT':
            model = ep.ThresholdModel(G, random_seed)
            for n in G.nodes():
                config.add_node_configuration("threshold", n, threshold)
        elif diff_type == 'IC':
            model = ep.IndependentCascadesModel(G, random_seed)
            for e in G.edges():
                config.add_edge_configuration("threshold", e, threshold)
        elif diff_type == 'SIS':
            model = ep.SISModel(G, random_seed)
            config.add_model_parameter('beta', infect_prob)
            config.add_model_parameter('lambda', recover_prob)
        elif diff_type == 'SIR':
            model = ep.SIRModel(G, random_seed)
            config.add_model_parameter('beta', infect_prob)
            config.add_model_parameter('gamma', recover_prob)
        elif diff_type == 'SI':
            model = ep.SIModel(G, random_seed)
            config.add_model_parameter('beta', infect_prob)
        else:
            raise ValueError('Only IC, LT, SI, SIR and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed_vector)
        model.set_initial_status(config)
        iterations = model.iteration_bunch(time_step)

        # Initialize node statuses
        current_status = {n: 0 for n in G.nodes()}

        for t in range(time_step):
            node_status = iterations[t]['status']
            inf_vec = np.array([current_status[n] for n in G.nodes()])

            for n, status in node_status.items():
                if status == 2:
                    status = 1  # Set recovered status to infected status (for SIR model)
                current_status[n] = status

            inf_vec = np.array([current_status[n] for n in G.nodes()])
            all_iterations.append(inf_vec)

        simulation[i, :, :] = np.array(all_iterations).T

    simulation = torch.Tensor(simulation)  # Shape: (sim_num, node_num, time_step)
    dataset = {'adj_mat': adj_mat, 'diff_mat': simulation}
    return dataset