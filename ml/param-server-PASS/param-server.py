import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import glob
import numpy as np
import torch.utils.data
from PIL import Image
from const import *
import ray
import sys

from ray.util.scheduling_strategies import (
    In,
    NotIn,
    Exists,
    DoesNotExist,
    NodeLabelSchedulingStrategy,
)

# MODEL = torchvision.models.resnet50()
# MODEL = torchvision.models.resnet18()
# MODEL = torchvision.models.mobilenet_v3_large()
MODEL =  torchvision.models.mobilenet_v3_small()

# DATA_DIR = "/home/ubuntu/train_ml_2/dataset_batch/"
# BATCH_SIZE = 50
# NODE_USER_NAME = "ubuntu"
# DATA_IP = "172.31.40.126"


use_scheduler = False

if len(sys.argv) > 1:
    mode = sys.argv[1]
    if mode == "sched":
        use_scheduler = True
        print("Using scheduler")
    else:
        print(f"Unknown mode: {mode}. Exiting")
        sys.exit(1)


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

def get_transform():
    custom_transforms = []
    
    custom_transforms.append(torchvision.transforms.Resize((400, 500)))
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

@ray.remote(num_cpus=1)
class ParameterServer(object):
    def __init__(self, lr):
        self.model = MODEL
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, gradients):
        self.optimizer.zero_grad()
        
        # self.model.set_gradients(summed_gradients)
        for g, p in zip(gradients, self.model.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g.copy())

        self.optimizer.step()

        # return self.model.get_weights()
        # return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}
    

@ray.remote(num_cpus=14)
def train_batch(working_dir, complexity_score, server):
    my_model = MODEL
    
    # my_model.set_weights(ray.get(server.get_weights.remote()))
    weights = ray.get(server.get_weights.remote())
    my_model.load_state_dict(weights)

    del weights

    my_model.zero_grad()

    my_dataset = torchvision.datasets.ImageFolder(root=working_dir, transform=get_transform())

    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE, # The whole folder is one batch
    )

    for imgs, labels in data_loader: # Only iterate once
        output = my_model(imgs)
        loss = F.nll_loss(output, labels)
    
    loss.backward()

    del my_dataset
    del data_loader

    grads = []
    for p in my_model.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)

    server.apply_gradients.remote(grads)

    print("A batch of training is done.")

    # return my_model.get_gradients()

@ray.remote(num_cpus=14)
def train_batch_manual(working_dir, complexity_score, server):
    if not os.path.exists(working_dir):
        remote = True
        # time.sleep(complexity_score / 100000)
        os.system(f"rsync -e 'ssh -o StrictHostKeyChecking=no' --mkpath -r -a {NODE_USER_NAME}@{DATA_IP}:{working_dir} {working_dir}")
    
    my_model = MODEL
    
    # my_model.set_weights(ray.get(server.get_weights.remote()))
    weights = ray.get(server.get_weights.remote())
    my_model.load_state_dict(weights)

    del weights

    my_model.zero_grad()

    my_dataset = torchvision.datasets.ImageFolder(root=working_dir, transform=get_transform())

    data_loader = torch.utils.data.DataLoader(
        my_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE, # The whole folder is one batch
    )

    for imgs, labels in data_loader: # Only iterate once
        output = my_model(imgs)
        loss = F.nll_loss(output, labels)
    
    loss.backward()

    del my_dataset
    del data_loader

    grads = []
    for p in my_model.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)

    server.apply_gradients.remote(grads)

    print("A batch of training is done.")



if __name__ == "__main__":
    ray.init(address="auto")
    # num_workers = 2

    data_batches = glob.glob(DATA_DIR)

    # server = ParameterServer.remote(1e-2)

    head_node_ip = os.getenv("HEAD_NODE_IP", None)
    if head_node_ip is None:
        raise Exception("env variable HEAD_NODE_IP not set, please tell us the head node ip.")

    nodes = ray.nodes()
    node_id = None
    for node in nodes:
        if node["NodeManagerAddress"] == head_node_ip:
            node_id = node["NodeID"]
            break

    server = ParameterServer.options(
            scheduling_strategy=NodeLabelSchedulingStrategy(
                hard={"ray.io/node_id": In(node_id)}
            )
        ).remote(1e-2)

    print("Running Asynchronous Parameter Server Training.")

    training_tasks = []

    start_time = time.time()

    for data in data_batches:
        cur_complexity = os.stat(data).st_size
        if use_scheduler:
            training_tasks.append(train_batch.remote(
                working_dir=data,
                complexity_score=cur_complexity,
                server=server
            ))
        else:
            training_tasks.append(train_batch_manual.remote(
                data,
                cur_complexity,
                server
            ))
    
    ray.get(training_tasks)

    total_time = time.time() - start_time

    # Test accuracy

    print(total_time)

    # gradients = {}

    # for i in range(iterations * num_workers):
    #     ready_gradient_list, _ = ray.wait(list(gradients))
    #     ready_gradient_id = ready_gradient_list[0]
    #     worker = gradients.pop(ready_gradient_id)

    #     # Compute and apply gradients.
    #     current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
    #     gradients[worker.compute_gradients.remote(current_weights)] = worker

        # if i % 10 == 0:
        #     # Evaluate the current model after every 10 updates.
        #     model.set_weights(ray.get(current_weights))
        #     accuracy = evaluate(model, test_loader)
        #     print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    # print("Final accuracy is {:.1f}.".format(accuracy))