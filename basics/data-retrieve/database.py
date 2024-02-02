import ray 
import time

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]

db_object_ref = ray.put(database)

@ray.remote
def retrieve_task(item, db):
    time.sleep(item / 10.)
    return item, db[item]

def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")

start = time.time()
object_references = [
    retrieve_task.remote(item, db_object_ref) for item in range(8)
]
all_data = []

while len(object_references) > 0:
    finished, object_references = ray.wait(
        object_references, timeout=7.0
    )
    data = ray.get(finished)
    print_runtime(data, start)
    all_data.extend(data)

print_runtime(all_data, start)