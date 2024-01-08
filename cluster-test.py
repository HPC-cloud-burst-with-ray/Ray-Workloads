from collections import Counter
import socket
import time

import ray

# ray.init(address='192.168.1.1:1234')
# ray.init(address='192.168.20.203:6379')     # do Ray
# ray.init(address='ray://192.168.1.3:10001') 
# ray.init(address=auto) 

ray.init(address="192.168.20.203:6379", _redis_password="YXHUILOPQ7")

# import time

# @ray.remote
# def f():
#     time.sleep(0.01)
#     return ray._private.services.get_node_ip_address()

# # Get a list of the IP addresses of the nodes that have joined the cluster.
# print(set(ray.get([f.remote() for _ in range(1000)])))


print('''This cluster consists of
    {} nodes in total and the following resources: \n
    {} 
'''.format(len(ray.nodes()), ray.cluster_resources()))

@ray.remote
def f():
    time.sleep(0.01)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(2000)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))

    
