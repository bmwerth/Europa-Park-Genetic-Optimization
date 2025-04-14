import matplotlib.pyplot as plt
import pandas as pd
import random
import math

POP_SIZE = 300  # Population size
MUTATION_RATE = 0.2  # Mutation rate
GENERATIONS = 300

# Create a fully connected graph of rides
ride_name_list = ['Alpine Express', 'Bumper Cars', 'Arthur', 'Atlantica SuperSplash', 'Atlantis Adventure','Blue Fire Megacoaster', 'Cassandras Curse', 'Castello dei Medici', 'Euro-Mir', 'Can Can Coaster', 'Fjord Rafting', 'Madame Freudenreich Curiosites', 'Matterhorn Blitz', 'Carousel', 'Pirates in Batavia', 'Queens Diamonds', 'Silver Star', 'Swiss Bob Run', 'Tirol Log Flume', 'Vindjammer', 'Voletarium', 'Voltron Nevera', 'Poseidon', 'Wodan']
ride_list = {
    0: {"name": "Alpine Express", "wait_time": 19, "duration": 2, "bryan_interest": 7, "dante_interest": 3, "pratik_interest": 1, "saahil_interest": 7},
    1: {"name": "Bumper Cars", "wait_time": 9, "duration": 2, "bryan_interest": 2, "dante_interest": 8, "pratik_interest": 7, "saahil_interest": 6},
    2: {"name": "Arthur", "wait_time": 31, "duration": 4, "bryan_interest": 9, "dante_interest": 2, "pratik_interest": 10, "saahil_interest": 6},
    3: {"name": "Atlantica SuperSplash", "wait_time": 18, "duration": 4, "bryan_interest": 5, "dante_interest": 1, "pratik_interest": 3, "saahil_interest": 3},
    4: {"name": "Atlantis Adventure", "wait_time": 7, "duration": 2, "bryan_interest": 8, "dante_interest": 4, "pratik_interest": 3, "saahil_interest": 4},
    5: {"name": "Blue Fire Megacoaster", "wait_time": 30, "duration": 3, "bryan_interest": 4, "dante_interest": 8, "pratik_interest": 8, "saahil_interest": 10},
    6: {"name": "Cassandras Curse", "wait_time": 6, "duration": 4, "bryan_interest": 5, "dante_interest": 10, "pratik_interest": 4, "saahil_interest": 9},
    7: {"name": "Castello dei Medici", "wait_time": 8, "duration": 4, "bryan_interest": 7.5, "dante_interest": 6, "pratik_interest": 7, "saahil_interest": 5},
    8: {"name": "Euro-Mir", "wait_time": 21, "duration": 5, "bryan_interest": 8, "dante_interest": 3, "pratik_interest": 1, "saahil_interest": 6.5},
    9: {"name": "Can Can Coaster", "wait_time": 30, "duration": 4, "bryan_interest": 9, "dante_interest": 6, "pratik_interest": 8, "saahil_interest": 3.},
    10: {"name": "Fjord Rafting", "wait_time": 21, "duration": 5, "bryan_interest": 7, "dante_interest": 4, "pratik_interest": 4, "saahil_interest": 4},
    11: {"name": "Madame Freudenreich Curiosites", "wait_time": 6, "duration": 6, "bryan_interest": 3, "dante_interest": 10, "pratik_interest": 1, "saahil_interest": 2},
    12: {"name": "Matterhorn Blitz", "wait_time": 25, "duration": 3, "bryan_interest": 7.5, "dante_interest": 2, "pratik_interest": 3, "saahil_interest": 8.5},
    13: {"name": "Carousel", "wait_time": 6, "duration": 3, "bryan_interest": 1, "dante_interest": 10, "pratik_interest": 1, "saahil_interest": 1},
    14: {"name": "Pirates in Batavia", "wait_time": 16, "duration": 8, "bryan_interest": 9, "dante_interest": 7, "pratik_interest": 10, "saahil_interest": 10},
    15: {"name": "Queens Diamonds", "wait_time": 17, "duration": 5, "bryan_interest": 9, "dante_interest": 5, "pratik_interest": 8, "saahil_interest": 7},
    16: {"name": "Silver Star", "wait_time": 26, "duration": 3, "bryan_interest": 2, "dante_interest": 6, "pratik_interest": 4, "saahil_interest": 4},
    17: {"name": "Swiss Bob Run", "wait_time": 25, "duration": 2, "bryan_interest": 10, "dante_interest": 3, "pratik_interest": 8, "saahil_interest": 6},
    18: {"name": "Tirol Log Flume", "wait_time": 14, "duration": 5, "bryan_interest": 8, "dante_interest": 2, "pratik_interest": 7, "saahil_interest": 7.5},
    19: {"name": "Vindjammer", "wait_time": 9, "duration": 3, "bryan_interest": 4, "dante_interest": 8, "pratik_interest": 6, "saahil_interest": 5},
    20: {"name": "Voletarium", "wait_time": 26, "duration": 5, "bryan_interest": 9, "dante_interest": 1, "pratik_interest": 8, "saahil_interest": 8.5},
    21: {"name": "Voltron Nevera", "wait_time": 49, "duration": 3, "bryan_interest": 2, "dante_interest": 1, "pratik_interest": 7, "saahil_interest": 4},
    22: {"name": "Poseidon", "wait_time": 19, "duration": 6, "bryan_interest": 8, "dante_interest": 2, "pratik_interest": 6, "saahil_interest": 4},
    23: {"name": "Wodan", "wait_time": 34, "duration": 4, "bryan_interest": 10, "dante_interest": 2, "pratik_interest": 7, "saahil_interest": 6}}

# Load the Excel file (replace 'your_file.xlsx' and 'Sheet1' with your actual file and sheet name)
df = pd.read_excel('Europa_Park_Model_v2_Data.xlsx', sheet_name='Sheet1')  # sheet_name is optional if it's the first sheet

# Convert to a 2D array (list of lists)
walk_times_list = df.values.tolist()

def choose_excluding_last(values, last_choice):
    choices = [v for v in values if v != last_choice]
    if not choices:
        raise ValueError("No values to choose from after excluding last choice.")
    return random.choice(choices)

def compute_fitness(route, rides, walk_times):
    total_time = rides[route[0]]["duration"] + rides[route[0]]["wait_time"]
    total_interest = (rides[route[0]]["bryan_interest"]+rides[route[0]]["dante_interest"]+rides[route[0]]["pratik_interest"]+rides[route[0]]["saahil_interest"])/4
    bryan_interest = rides[route[0]]["bryan_interest"]
    dante_interest = rides[route[0]]["dante_interest"]
    pratik_interest = rides[route[0]]["pratik_interest"]
    saahil_interest = rides[route[0]]["saahil_interest"]
    i = 1
    while i < len(route):
        if(rides[route[i]]["name"] == rides[route[i-1]]["name"]):
            return -1, -1, -1, -1, -1, -1
        total_time += walk_times[route[i-1]][route[i]+1] + rides[route[i]]["duration"] + rides[route[i]]["wait_time"]
        total_interest += (rides[route[i]]["bryan_interest"]+rides[route[i]]["dante_interest"]+rides[route[i]]["pratik_interest"]+rides[route[i]]["saahil_interest"])/4
        bryan_interest += rides[route[i]]["bryan_interest"]
        dante_interest += rides[route[i]]["dante_interest"]
        pratik_interest += rides[route[i]]["pratik_interest"]
        saahil_interest += rides[route[i]]["saahil_interest"]
        i += 1
    if total_time > 540:
        return -1, -1, -1, -1, -1, -1
    return (total_interest*.5+min(bryan_interest, dante_interest, pratik_interest, saahil_interest)*.5), i, bryan_interest/i, dante_interest/i, pratik_interest/i, saahil_interest/i

def generate_random_route(rides, walk_times):
    route = []
    i = 0
    current = random.choice(list(rides.keys()))
    time = rides[current]["duration"] + rides[current]["wait_time"]
    route.append(current)
    while True:
        ride = choose_excluding_last(list(rides.keys()), current)
        travel = walk_times[current][ride+1]
        duration = rides[ride]["duration"]
        wait_time = rides[ride]["wait_time"]
        if time + travel + duration + wait_time > 540:
            break
        route.append(ride)
        time = time + travel + duration + wait_time
        current = ride
    return route

def crossover(route1, route2):
    cut = random.randint(1, min(len(route1), len(route2)) - 1)
    return route1[:cut] + route2[cut:]

def mutate(route, rides):
    if len(route) == 0:
        return route
    i = random.randint(0, len(route)-1)
    new_ride = random.choice(list(rides.keys()))
    route[i] = new_ride
    return route

def select_parents(population, fitnesses):
    total = sum(fitnesses)
    weights = [f / total if (total>0) else 1 / len(fitnesses) for f in fitnesses]
    return random.choices(population, weights=weights, k=2)

def next_generation(population):
    fitnesses = [compute_fitness(r, ride_list, walk_times_list)[0] for r in population]
    new_pop = []
    while len(new_pop) < POP_SIZE:
        p1, p2 = select_parents(population, fitnesses)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child,ride_list)
        new_pop.append(child)
    return new_pop

def run_evolution():
    # Turn on interactive mode
    plt.ion()

    # Create the figure and axis
    fig, ax = plt.subplots()
    x_data, y_data = [], []
    sc = ax.scatter(x_data, y_data)
    
    population = [generate_random_route(ride_list,walk_times_list) for _ in range(POP_SIZE)]
    best_route = None
    best_data = None
    generation_num = 1
    for _ in range(GENERATIONS):
        population = next_generation(population)
        # Create a list of (candidate, fitness_tuple) pairs
        fitness_results = [(r, compute_fitness(r, ride_list, walk_times_list)) for r in population]
        # Find the one with the best fitness (first value in the tuple)
        best_candidate, best_fitness_tuple = max(fitness_results, key=lambda x: x[1][0])
        # Update best if it's None or if the current candidate is better
        if best_route is None or best_fitness_tuple[0] > best_data[0]:
            best_route, best_data = best_candidate, best_fitness_tuple
        
        # Add to the data lists
        x_data.append(generation_num)
        y_data.append(best_fitness_tuple[0])

        # Update scatter data
        sc.remove()  # remove old scatter
        sc = ax.scatter(x_data, y_data, color='blue')

        # Redraw
        plt.pause(0.1)  # Small pause to allow GUI to update
        generation_num += 1
    return best_route, best_data

optimized_route, optimized_data = run_evolution()
print(optimized_route, optimized_data[0])
print("Bryan's Average Interest is ", optimized_data[2])
print("Dante's Average Interest is ", optimized_data[3])
print("Pratik's Average Interest is ", optimized_data[4])
print("Saahil's Average Interest is ", optimized_data[5])

# Assign custom distances and interest
#for u, v in G.edges():
#    start_index = ride_name_list.index(u)
#    end_index = ride_name_list.index(v)
#    distance = walk_times_list[start_index][end_index+1]  # e.g., walking time in seconds or meters
#    print(u, v, start_index, end_index, distance)


