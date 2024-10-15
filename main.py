import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import torch.optim as optim
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
from utils import load_dataset

# setting the device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VPSLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate1, dropout_rate2):
        super(VPSLModel, self).__init__()

        # VOTING MODEL
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate1)
        self.fc2 = nn.Linear(hidden_dim1, input_dim)

        #  DETERMINATION MODEL
        self.fc3 = nn.Linear(input_dim, hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate2)
        self.fc4 = nn.Linear(hidden_dim2, 1)

    def forward(self, sample_matrix, sample_mask, nodes, adj_list):
        max_len = sample_matrix.size(1)
        middle_matrices = []

        for n, node in enumerate(nodes[:, 0, 0]):
            i = 0
            neighbors_data = sample_matrix[n, i, :].unsqueeze(0)
            mask = sample_mask[n, i, :].unsqueeze(0)
            x = F.relu(self.fc1(neighbors_data))
            x = self.dropout1(x)
            output = self.fc2(x)
            output[mask == 0] = float('-inf')  # making sure the probability non-active nodes to 0
            output = torch.softmax(output, dim=1)
            middle_matrix = torch.full((1, max_len), 0, dtype=torch.float32).to(sample_matrix.device)
            middle_matrix[0, 0] = output[0, 0]

            for numi, neighbornode in enumerate(adj_list[node.item()]):
                i += 1
                nneighbors_data = sample_matrix[n, i, :].unsqueeze(0)
                nmask = sample_mask[n, i, :].unsqueeze(0)
                nx = F.relu(self.fc1(nneighbors_data))
                nx = self.dropout1(nx)
                noutput = self.fc2(nx)
                noutput[nmask == 0] = float('-inf')  # making sure the probability non-active nodes to 0
                noutput = torch.softmax(noutput, dim=1)
                node_positionsidx = find_node_pair_positions(adj_list, neighbornode, node.item())
                node_positions = node_positionsidx[0] + 1
                middle_matrix[0, numi + 1] = noutput[0, node_positions]

            middle_matrices.append(middle_matrix)

        middle_matrices = torch.cat(middle_matrices, dim=0)
        middle_matrices = torch.where(torch.isnan(middle_matrices), torch.tensor(0.0, device=middle_matrices.device),
                                      middle_matrices)

        # using the  DETERMINATION MODEL
        y = F.relu(self.fc3(middle_matrices))
        y = self.dropout2(y)
        routput = self.fc4(y)
        routput = torch.sigmoid(routput)
        routput = routput.reshape(-1)
        return routput


def find_node_value(sample_node_all, val0, val1):
    for node_data in sample_node_all:
        # find the node position from the list of [val0, val1, val2]  val0:page of dataset; val1:round; val2:node
        if node_data[0, 0] == val0 and node_data[0, 1] == val1:
            return node_data[0, 2]


# FINE-TURN MODEL
def train_model(params):
    final_model = VPSLModel(input_dim=inner, hidden_dim1=params['hidden_dim1'],
                            hidden_dim2=params['hidden_dim2'], output_dim=1,
                            dropout_rate1=params['dropout_rate1'],
                            dropout_rate2=params['dropout_rate2']).to(DEVICE)

    optimizer = optim.Adam(final_model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    scaler = torch.amp.GradScaler()
    global best_val_loss
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(params['num_epochs']):
        total_loss = 0
        final_model.train()
        for batch in train_loader:
            nodes = batch
            sample_matrix = np.full((batch.size(0), inner, inner), pal, dtype=np.int16)
            sample_mask = np.zeros((batch.size(0), inner, inner), dtype=np.int8)
            labels = np.zeros((1, batch.size(0)))

            for i in range(batch.size(0)):
                sample_matrix[i, 0, :] = sample_matrix_all[int(nodes[i, 0, 2].item())]
                sample_mask[i, 0, :] = sample_mask_all[int(nodes[i, 0, 2].item())]
                round = nodes[i, 0, 1]
                node = nodes[i, 0, 0]
                labels[0, i] = X[int(nodes[i, 0, 0]), int(round + 1)]
                j = 1
                for j, neighbornode in enumerate(adj_list[int(node)], start=1):
                    idx = find_node_value(sample_node_all, neighbornode, round)
                    if idx == None:
                        j -= 1
                        continue
                    sample_matrix[i, j, :] = sample_matrix_all[int(idx)]
                    sample_mask[i, j, :] = sample_mask_all[int(idx)]

            sample_matrix = torch.from_numpy(sample_matrix).float().to(DEVICE)
            labels = torch.from_numpy(labels).float().to(DEVICE).reshape(-1)
            sample_mask = torch.from_numpy(sample_mask).float().to(DEVICE)
            nodes = nodes.to(DEVICE)

            routput = final_model(sample_matrix, sample_mask, nodes, adj_list)
            loss = criterion(routput, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            del sample_matrix, sample_mask, labels

        scheduler.step()
        val_loss = validate(final_model, val_loader, criterion)
        print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params['patience']:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    return final_model, optimizer


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            nodes = batch
            sample_matrix = np.full((batch.size(0), inner, inner), pal, dtype=np.int16)
            sample_mask = np.zeros((batch.size(0), inner, inner), dtype=np.int8)
            labels = np.zeros((1, batch.size(0)))

            for i in range(batch.size(0)):
                sample_matrix[i, 0, :] = sample_matrix_all[int(nodes[i, 0, 2].item())]
                sample_mask[i, 0, :] = sample_mask_all[int(nodes[i, 0, 2].item())]
                round = nodes[i, 0, 1]
                labels[0, i] = X[int(nodes[i, 0, 0]), int(round + 1)]
                node = nodes[i, 0, 0]
                j = 1
                for j, neighbornode in enumerate(adj_list[int(node)], start=1):
                    idx = find_node_value(sample_node_all, neighbornode, round)
                    if idx == None:
                        j -= 1
                        continue
                    sample_matrix[i, j, :] = sample_matrix_all[int(idx)]
                    sample_mask[i, j, :] = sample_mask_all[int(idx)]

            sample_matrix = torch.from_numpy(sample_matrix).float().to(DEVICE)
            labels = torch.from_numpy(labels).float().to(DEVICE).reshape(-1)
            sample_mask = torch.from_numpy(sample_mask).float().to(DEVICE)
            nodes = nodes.to(DEVICE)

            routput = model(sample_matrix, sample_mask, nodes, adj_list)
            loss = criterion(routput, labels)
            val_loss += loss.item()
            del sample_matrix, sample_mask, labels

    return val_loss / len(val_loader)


# 进行适配更改
class Sample:
    def __init__(self, sample_node):
        self.sample_node = sample_node


class CustomDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample.sample_node  # Accessing the attribute of the custom object


def get_max_neighbors(adj_list):
    max_neighbors = 0
    for node, neighbors in adj_list.items():
        num_neighbors = len(neighbors)
        if num_neighbors > max_neighbors:
            max_neighbors = num_neighbors
    return max_neighbors


def get_neighbors_data(node, adj_list, round, max_len, pal, X):
    neighbors = [node]
    neighbors.extend(adj_list[node])
    neighbors_data = X[neighbors, round].detach().cpu().numpy()
    if len(neighbors_data) < max_len:
        neighbors_data = np.pad(neighbors_data, (0, max_len - len(neighbors_data)), 'constant', constant_values=pal)
    elif len(neighbors_data) > max_len:
        neighbors_data = neighbors_data[:max_len]
    neighbors_data = torch.tensor(neighbors_data, dtype=torch.float32).unsqueeze(0)
    return neighbors_data


def get_neighbors_data_test(node, adj_list, max_len, pal, Y):
    neighbors = [node]
    neighbors.extend(adj_list[node])
    neighbors_data = Y[neighbors].detach().cpu().numpy()
    if len(neighbors_data) < max_len:
        neighbors_data = np.pad(neighbors_data, (0, max_len - len(neighbors_data)), 'constant', constant_values=pal)
    elif len(neighbors_data) > max_len:
        neighbors_data = neighbors_data[:max_len]
    neighbors_data = torch.tensor(neighbors_data, dtype=torch.float32).unsqueeze(0)
    return neighbors_data


def create_antidiagonal_matrix(size):
    mat = np.zeros((size, size))
    np.fill_diagonal(np.fliplr(mat), 1)
    return torch.tensor(mat, dtype=torch.float32)


def find_node_pair_positions(adj_list, neighbornode, node):
    positions = []
    for idx, neighbor in enumerate(adj_list[neighbornode]):
        if neighbor == node:
            positions.append(idx)
            break
    return positions


def process_node_data(node, adj_list, round, max_len, pal, X, i, sample_matrix, sample_mask):
    self_state = X[node, round].unsqueeze(0)
    neighbors_data = get_neighbors_data(node, adj_list, round, max_len - 1, pal, X)
    mask = torch.ones_like(neighbors_data)
    mask[neighbors_data == pal] = 0
    sample_matrix[i, 0] = self_state
    sample_matrix[i, 1:neighbors_data.shape[1] + 1] = neighbors_data
    sample_mask[i, 0] = 0 if self_state == pal else 1
    sample_mask[i, 1:mask.shape[1] + 1] = mask
    return len(adj_list[node])


def process_node_data_test(node, adj_list, max_len, pal, Y, i, sample_matrix, sample_mask):
    self_state = Y[node].unsqueeze(0)
    neighbors_data = get_neighbors_data_test(node, adj_list, max_len - 1, pal, Y)
    mask = torch.ones_like(neighbors_data)
    mask[neighbors_data == pal] = 0
    sample_matrix[i, 0] = self_state
    sample_matrix[i, 1:neighbors_data.shape[1] + 1] = neighbors_data
    sample_mask[i, 0] = 0 if self_state == pal else 1
    sample_mask[i, 1:mask.shape[1] + 1] = mask
    return len(adj_list[node])


def load_pretrained_model(model_save_path):
    checkpoint = torch.load(model_save_path, weights_only=True)
    params = checkpoint['params']
    input_dim = checkpoint['input_dim']  # Load the input dimension
    model = VPSLModel(input_dim=input_dim,  # Use the saved input dimension
                      hidden_dim1=params['hidden_dim1'],
                      hidden_dim2=params['hidden_dim2'],
                      output_dim=1,
                      dropout_rate1=params['dropout_rate1'],
                      dropout_rate2=params['dropout_rate2']).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def get_top_k_binary_predictions(output_tensor, k):
    sorted_tensor = output_tensor.clone()
    top_k_indices = torch.topk(sorted_tensor, k).indices
    binary_predictions = torch.zeros_like(output_tensor)
    binary_predictions[top_k_indices] = 1
    return binary_predictions


def label_samples(samples):
    labels = []
    for sample in samples:
        if sample.sample_matrix[0][0] == 1 and sample.label == 1:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(labels)


#Fine-Turning
def fine_tune_model(model, fine_tune_loader, criterion, optimizer, num_fine_tune_epochs=5):
    model.train()
    for epoch in range(num_fine_tune_epochs):
        for batch in fine_tune_loader:
            nodes = batch
            sample_matrix = np.full((batch.size(0), inner, inner), pal, dtype=np.int16)
            sample_mask = np.zeros((batch.size(0), inner, inner), dtype=np.int8)
            labels = np.zeros((1, batch.size(0)))
            for i in range(batch.size(0)):
                sample_matrix[i, 0, :] = f_sample_matrix_all[int(nodes[i, 0, 2].item())]
                sample_mask[i, 0, :] = f_sample_mask_all[int(nodes[i, 0, 2].item())]
                round = nodes[i, 0, 1]
                labels[0, i] = X[int(nodes[i, 0, 0]), int(round + 1)]
                node = nodes[i, 0, 0]
                j = 1
                for j, neighbornode in enumerate(adj_list[int(node)], start=1):
                    idx = find_node_value(f_sample_node_all, neighbornode, round)
                    if idx == None:
                        j -= 1
                        continue
                    sample_matrix[i, j, :] = f_sample_matrix_all[int(idx)]
                    sample_mask[i, j, :] = f_sample_mask_all[int(idx)]

            sample_matrix = torch.from_numpy(sample_matrix).float().to(DEVICE)
            labels = torch.from_numpy(labels).float().to(DEVICE).reshape(-1)
            sample_mask = torch.from_numpy(sample_mask).float().to(DEVICE)
            nodes = nodes.to(DEVICE)

            output = model(sample_matrix, sample_mask, nodes, adj_list)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# load datasets ('karate', 'dolphins', 'jazz', 'netscience',  'cora_ml')
curr_dir = os.getcwd()
data_name = 'karate'  # change the name through the 'karate', 'dolphins', 'jazz', 'netscience',  'cora_ml'
graph = load_dataset(data_name, data_dir=curr_dir)
A = graph['adj_mat']
A = A.toarray()
A = torch.tensor(A, dtype=torch.float32)

num_nodes = A.shape[0]
adj_list = {i: [] for i in range(num_nodes)}
rows, cols = np.where(A == 1)
for i, j in zip(rows, cols):
    adj_list[i].append(j)

train_data = np.load("diff_mat/diff_mat_train_karate.npy")  # make sure the name suits with the graph you choose
test_data = np.load("diff_mat/diff_mat_test_karate.npy")  # make sure the name suits with the graph you choose

total_pages = train_data.shape[0]
acc_list = []
auc_list = []
pr_list = []
re_list = []
f1_list = []
confusion_matrices = []

# Training parameters
params = {
    'hidden_dim1': 128,
    'hidden_dim2': 128,
    'dropout_rate1': 0.7,
    'dropout_rate2': 0.5,
    'learning_rate': 1e-5,
    'num_epochs': 20,
    'patience': 10
}

# Preparing the Pre-train dataset
all_samples = []
sample_matrix_all = []
sample_mask_all = []
sample_node_all = []
for page in range(1):
    data_extracted = train_data[page]
    counts = np.sum(data_extracted == 1, axis=0)
    proportions_dict = {f'column_{i}': count / num_nodes for i, count in enumerate(counts)}
    filtered_positions = [i for i, proportion in enumerate(proportions_dict.values()) if
                          0.3 < proportion < 0.8]  # you can choose the scale by yourself in different model as the percentage number
    start = filtered_positions[0]
    end = filtered_positions[-1]
    data_set = train_data[page, :, start:end + 1]
    true_labels = train_data[page, :, start - 1]
    max_neighbors = get_max_neighbors(adj_list)
    inner = max_neighbors + 2
    X = torch.tensor(data_set[:, :], dtype=torch.float32).to(DEVICE)
    antidiagonal_matrix = create_antidiagonal_matrix(end - start + 1).to(DEVICE)
    X = torch.matmul(X, antidiagonal_matrix)
    max_len = inner
    pal = 0
    count = 0
    for round in range(end - start):
        for node in range(num_nodes):
            if X[node, round] != 0 and len(adj_list[node]) != 1:
                sample_matrix = np.full((1, inner), pal, dtype=np.int16)
                sample_mask = np.zeros((1, inner), dtype=np.int8)
                sample_node = np.zeros((1, 3))
                sample_matrix[0, -1] = process_node_data(node, adj_list, round, max_len, pal, X, 0, sample_matrix,
                                                         sample_mask)
                sample_node[0, 0] = int(node)
                sample_node[0, 1] = int(round)
                sample_node[0, 2] = int(count)
                sample_matrix_all.append(sample_matrix)
                sample_mask_all.append(sample_mask)
                sample_node_all.append(sample_node)
                all_samples.append(Sample(sample_node))
                count += 1

dataset = CustomDataset(all_samples)

# Depart the dataset to the training and the test dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

criterion = BCEWithLogitsLoss()
final_model, optimizer = train_model(params)

# saving the pre-trained model
model_save_path = 'final_model_karate.pth' #making sure the suffix suits with the graph name
torch.save({
    'model_state_dict': final_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'params': params,
    'input_dim': inner
}, model_save_path)
print(f'Model saved to {model_save_path}')

all_samples.clear()

# Using part of the model
total_pages = test_data.shape[0]

for page in range(total_pages):
    # Load pre-trained model
    model_save_path = 'final_model_karate.pth' #making sure the suffix suits with the graph name
    model, optimizer = load_pretrained_model(model_save_path)

    # Prepare data for fine-tuning
    fine_tune_samples = []
    f_sample_matrix_all = []
    f_sample_mask_all = []
    f_sample_node_all = []
    count = 0

    # preparing the fine-turn dataset
    data_extracted = test_data[page]
    counts = np.sum(data_extracted == 1, axis=0)
    proportions_dict = {f'column_{i}': count / num_nodes for i, count in enumerate(counts)}
    filtered_positions = [i for i, proportion in enumerate(proportions_dict.values()) if
                          0.15 < proportion < 0.7]  # you can choose the scale by yourself in different model as the percentage number
    start = filtered_positions[0]
    end = start + 3 # t
    if end > 24:
        end = 24
    data_set = test_data[page, :, start:end + 1]
    max_neighbors = get_max_neighbors(adj_list)
    inner = max_neighbors + 2
    X = torch.tensor(data_set[:, :], dtype=torch.float32).to(DEVICE)
    antidiagonal_matrix = create_antidiagonal_matrix(end - start + 1).to(DEVICE)
    X = torch.matmul(X, antidiagonal_matrix)
    max_len = inner
    pal = 0
    for round in range(end - start):
        for node in range(num_nodes):
            if X[node, round] != 0 and len(adj_list[node]) != 1:
                sample_matrix = np.full((1, inner), pal, dtype=np.int16)
                sample_mask = np.zeros((1, inner), dtype=np.int8)
                sample_node = np.zeros((1, 3))
                sample_matrix[0, -1] = process_node_data(node, adj_list, round, max_len, pal, X, 0, sample_matrix,
                                                         sample_mask)
                sample_node[0, 0] = int(node)
                sample_node[0, 1] = int(round)
                sample_node[0, 2] = int(count)
                f_sample_matrix_all.append(sample_matrix)
                f_sample_mask_all.append(sample_mask)
                f_sample_node_all.append(sample_node)
                fine_tune_samples.append(Sample(sample_node))
                count += 1

    fine_tune_dataset = CustomDataset(fine_tune_samples)
    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=32, shuffle=True)
    criterion = BCEWithLogitsLoss()
    fine_tune_model(model, fine_tune_loader, criterion, optimizer, num_fine_tune_epochs=10)

    #the using part of the model
    model.eval()
    final_round = X[:, -1].clone().detach().to(DEVICE)
    output_tensor = torch.zeros(final_round.size(0), device=DEVICE)
    tolerance = 0
    max_iterations = 10
    previous_output = None
    stabilized = False
    iteration = 0

    while not stabilized and iteration < max_iterations:
        iteration += 1
        previous_output = final_round.clone()
        Y = previous_output

        test_samples = []
        t_sample_matrix_all = []
        t_sample_mask_all = []
        t_sample_node_all = []
        count = 0
        nodes_to_update = []  # 存储需要更新的节点
        for node in range(num_nodes):
            if previous_output[node] != 0 and len(adj_list[node]) != 1:
                sample_matrix = np.full((1, inner), pal, dtype=np.int16)
                sample_mask = np.zeros((1, inner), dtype=np.int8)
                sample_node = np.zeros((1, 3))
                sample_matrix[0, -1] = process_node_data_test(node, adj_list, max_len, pal, Y, 0, sample_matrix,
                                                              sample_mask)
                sample_node[0, 0] = int(node)
                sample_node[0, 1] = int(round)
                sample_node[0, 2] = int(count)
                t_sample_matrix_all.append(sample_matrix)
                t_sample_mask_all.append(sample_mask)
                t_sample_node_all.append(sample_node)
                test_samples.append(Sample(sample_node))
                count += 1
                nodes_to_update.append(node)  # 记录需要更新的节点

        test_dataset = CustomDataset(test_samples)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            predictions = previous_output.clone()  # 初始化predictions为previous_output
            for batch in test_loader:
                nodes = batch
                sample_matrix = np.full((batch.size(0), inner, inner), pal, dtype=np.int16)
                sample_mask = np.zeros((batch.size(0), inner, inner), dtype=np.int8)
                labels = np.zeros((1, batch.size(0)))
                for i in range(batch.size(0)):
                    sample_matrix[i, 0, :] = t_sample_matrix_all[int(nodes[i, 0, 2].item())]
                    sample_mask[i, 0, :] = t_sample_mask_all[int(nodes[i, 0, 2].item())]
                    round = nodes[i, 0, 1]
                    # labels[0, i] = X[int(nodes[i, 0, 0]), int(round + 1)]
                    node = nodes[i, 0, 0]
                    j = 1
                    for j, neighbornode in enumerate(adj_list[int(node)], start=1):
                        idx = find_node_value(t_sample_node_all, neighbornode, round)
                        if idx == None:
                            j -= 1
                            continue
                        sample_matrix[i, j, :] = t_sample_matrix_all[int(idx)]
                        sample_mask[i, j, :] = t_sample_mask_all[int(idx)]

                sample_matrix = torch.from_numpy(sample_matrix).float().to(DEVICE)
                # labels = torch.from_numpy(labels).float().to(DEVICE).reshape(-1)
                sample_mask = torch.from_numpy(sample_mask).float().to(DEVICE)
                nodes = nodes.to(DEVICE)

                output = model(sample_matrix, sample_mask, nodes, adj_list)
                for idx, node in enumerate(nodes[:, 0, 0]):
                    predictions[int(node.item())] = output[idx]  # 只更新符合条件的节点

        final_round = predictions

        if (predictions == 0).sum().item() > int(num_nodes * 0.9):
            final_round = previous_output  # 维持上一个输出
            break

        if torch.allclose(final_round, previous_output, atol=tolerance):
            stabilized = True

    print(f"Final output after {iteration} iterations")

    true_labels_eval = test_data[page, :, 0]
    k = int(num_nodes * 0.1)
    binary_predictions = get_top_k_binary_predictions(final_round, k)
    binary_predictions_cpu = binary_predictions.cpu().numpy()
    true_labels = test_data[page, :, 0]
    true_labels = torch.from_numpy(true_labels).float().to(DEVICE)
    acc = accuracy_score(true_labels.cpu().detach(), binary_predictions.cpu().detach())
    pr = precision_score(true_labels.cpu().detach(), binary_predictions.cpu().detach(), average='macro',
                         zero_division=0)
    re = recall_score(true_labels.cpu().detach(), binary_predictions.cpu().detach(), average='macro', zero_division=0)
    f1 = f1_score(true_labels.cpu().detach(), binary_predictions.cpu().detach(), average='macro', zero_division=0)
    auc = roc_auc_score(true_labels.cpu().detach(), predictions.cpu().detach())
    conf_matrix = confusion_matrix(true_labels.cpu(), binary_predictions.cpu())

    acc_list.append(acc)
    auc_list.append(auc)
    pr_list.append(pr)
    re_list.append(re)
    f1_list.append(f1)
    confusion_matrices.append(conf_matrix)

final_acc = sum(acc_list) / len(acc_list)
final_auc = sum(auc_list) / len(auc_list)
final_pr = sum(pr_list) / len(pr_list)
final_re = sum(re_list) / len(re_list)
final_f1 = sum(f1_list) / len(f1_list)

print("VPSL")
print(f"Final Accuracy: {final_acc:.3f}")
print(f"Final Precision: {final_pr:.3f}")
print(f"Final Recall: {final_re:.3f}")
print(f"Final F1-score: {final_f1:.3f}")
print(f"Final AUC: {final_auc:.3f}")

for i, conf_matrix in enumerate(confusion_matrices):
    TN, FP = conf_matrix[0][0], conf_matrix[0][1]
    FN, TP = conf_matrix[1][0], conf_matrix[1][1]
    print(f"Experiment {i + 1}:")
    print(f"  True Positive (TP): {TP}")
    print(f"  True Negative (TN): {TN}")
    print(f"  False Positive (FP): {FP}")
    print(f"  False Negative (FN): {FN}\n")
