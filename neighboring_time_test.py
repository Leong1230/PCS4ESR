from pytorch3d.ops import knn_points
from hybridpc.utils.serialization import encode
import MinkowskiEngine as ME
import torch
import time

SerialOrders = ["z", "z-trans", "hilbert", "hilbert-trans"]
KNeighbors = 8

def serial_neighbor(points, query_xyz):
    all_neighbor_idx = []
    grid_size = 0.01
    sort_queries = False
    quantization_time = 0.0
    encode_time = 0.0
    input_sorting_time = 0.0
    query_insertion_time = 0.0
    neighbor_indices_time = 0.0
    # for order in self.serial_orders:
    order = SerialOrders[0]

    quantization_time = time.time()
    query_coords, index, inverse_index = ME.utils.sparse_quantize(coordinates=query_xyz, return_index=True, return_inverse=True, quantization_size=(grid_size), device=query_xyz.device)
    # source_coords, source_index, source_inverse_index = ME.utils.sparse_quantize(coordinates=points, return_index=True, return_inverse=True, quantization_size=(grid_size), device=points.device)
    source_coords = torch.floor(points/grid_size).to(torch.int)
    quantization_time = time.time() - quantization_time

    torch.cuda.empty_cache()
    encode_time = time.time()
    depth = int(query_coords.max()).bit_length()
    query_codes = encode(query_coords, torch.zeros(query_coords.shape[0], dtype=torch.int64, device=query_coords.device), depth, order=order)
    source_codes = encode(source_coords, torch.zeros(source_coords.shape[0], dtype=torch.int64, device=source_coords.device), depth, order=order)
    encode_time = time.time() - encode_time

    torch.cuda.empty_cache()
    input_sorting_time = time.time()
    sorted_source_codes, sorted_source_indices = torch.sort(source_codes)
    input_sorting_time = time.time() - input_sorting_time

    torch.cuda.empty_cache()
    query_insertion_time = time.time()
    if sort_queries:
        sorted_query_codes, sorted_query_indices = torch.sort(query_codes)
        nearest_right_positions = torch.searchsorted(sorted_source_codes, sorted_query_codes, right=True)
    else:
        nearest_right_positions = torch.searchsorted(sorted_source_codes, query_codes, right=True)

    k = int(KNeighbors/2)  # Number of neighbors in each direction
    front_indices = nearest_right_positions.unsqueeze(1) - torch.arange(1, k+1).to(nearest_right_positions.device).unsqueeze(0)
    back_indices = nearest_right_positions.unsqueeze(1) + torch.arange(0, k).to(nearest_right_positions.device).unsqueeze(0)
    query_insertion_time = time.time() - query_insertion_time

    neighbor_indices_time = time.time()
    # Combine front and back indices
    neighbor_indices = torch.cat((front_indices, back_indices), dim=1)
    # Pad indices that are out of range by -1
    neighbor_indices = torch.where((neighbor_indices >= 0) & (neighbor_indices < len(sorted_source_codes)), neighbor_indices, torch.tensor(-1))

    # Map the indices back to the original unsorted source codes
    neighbor_source_indices = torch.where(neighbor_indices != -1, sorted_source_indices[neighbor_indices], torch.tensor(-1))

    # Reorder the neighbors to match the original order of the query codes
    if sort_queries:
        neighbor_idx = neighbor_source_indices[torch.argsort(sorted_query_indices)]
    else:
        neighbor_idx = neighbor_source_indices

    if len(inverse_index) > 0:
        neighbor_idx = neighbor_idx[inverse_index]

    all_neighbor_idx.append(neighbor_idx)
    neighbor_indices_time = time.time() - neighbor_indices_time

    return quantization_time, encode_time, input_sorting_time, query_insertion_time, neighbor_indices_time

def knn_neighbor(points, query_xyz):
    knn_output = knn_points(query_xyz.unsqueeze(0),
                            points.unsqueeze(0),
                            K=KNeighbors)
    
    all_neighbor_idx = knn_output.idx.squeeze(0)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Constants
    input_num_range = [1000, 5000, 25000, 125000, 625000]
    query_num_range = [1000, 5000, 25000, 125000, 625000, 3125000]
    
    # Record times
    knn_times = []
    serial_times = []

    # # First group: Fixed input points, varying query points
    # for input_num in input_num_range: 
    #     for query_num in query_num_range:
    #         torch.cuda.empty_cache()
    #         input_xyz = torch.rand((input_num, 3), device=device) * 2 - 1
    #         query_xyz = torch.rand((query_num, 3), device=device) * 2 - 1
            
    #         # Measure time for KNN
    #         knn_neighbor(input_xyz, query_xyz)
    #         start_time = time.time()
    #         knn_neighbor(input_xyz, query_xyz)
    #         knn_time = time.time() - start_time
    #         knn_times.append((input_num, query_num, knn_time))
            
    #         # Measure time for Serial Neighbor
    #         serial_neighbor(input_xyz, query_xyz)
    #         start_time = time.time()
    #         serial_neighbor(input_xyz, query_xyz)
    #         serial_time = time.time() - start_time
    #         serial_times.append((input_num, query_num, serial_time))

    # # Printing results in table format
    # print("Results for varying both input and query points:")
    # print(f"{'Input Points':<15}{'Query Points':<15}{'KNN Time (s)':<15}{'Serial Time (s)':<15}")
    # print("-" * 60)
    # for (input_num, query_num, knn_time), (_, _, serial_time) in zip(knn_times, serial_times):
    #     print(f"{input_num:<15}{query_num:<15}{knn_time:<15.6f}{serial_time:<15.6f}")

    # Constants
    input_num_range = [5000, 25000, 125000, 625000]
    query_num_range = [25000, 125000, 625000, 3125000]
    
    # Record times
    knn_times = []
    serial_times = []

    # First group: Fixed input points, varying query points
    for input_num in input_num_range: 
        for query_num in query_num_range:
            torch.cuda.empty_cache()
            input_xyz = torch.rand((input_num, 3), device=device) * 2 - 1
            query_xyz = torch.rand((query_num, 3), device=device) * 2 - 1
            
            # # Measure time for KNN
            # knn_neighbor(input_xyz, query_xyz)
            # start_time = time.time()
            # knn_neighbor(input_xyz, query_xyz)
            # knn_time = time.time() - start_time
            # knn_times.append((input_num, query_num, knn_time))
            
            # Measure time for Serial Neighbor
            serial_neighbor(input_xyz, query_xyz)
            start_time = time.time()
            quantization_time, encode_time, input_sorting_time, query_insertion_time, neighbor_indices_time = serial_neighbor(input_xyz, query_xyz)
            serial_time = time.time() - start_time
            serial_times.append((input_num, query_num, serial_time, quantization_time, encode_time, input_sorting_time, query_insertion_time, neighbor_indices_time))

    # Printing results in detail
    # Printing results in detail
    print("Detailed times for each component in serialization (in ms):")
    print(f"{'Input Points':<15}{'Query Points':<15}{'Serial (ms)':<15}{'Quantization (ms)':<20}{'Encoding (ms)':<15}{'Sorting (ms)':<15}{'Insertion (ms)':<15}{'Neighbor Indices (ms)':<20}")
    print("-" * 120)
    for entry in serial_times:
        input_num, query_num, serial_time, quantization_time, encode_time, input_sorting_time, query_insertion_time, neighbor_indices_time = entry
        print(f"{input_num:<15}{query_num:<15}{serial_time * 1000:<15.6f}{quantization_time * 1000:<20.6f}{encode_time * 1000:<15.6f}{input_sorting_time * 1000:<15.6f}{query_insertion_time * 1000:<15.6f}{neighbor_indices_time * 1000:<20.6f}")



    # # Second group: Fixed query points, varying input points
    # query_num_fixed = 10000  # Fixed number of query points
    # input_num_range = [1000, 10000, 100000, 1000000]  # Different numbers of input points
    
    # for input_num in input_num_range:
    #     input_xyz = torch.rand((input_num, 3), device=device) * 2 - 1
    #     query_xyz = torch.rand((query_num_fixed, 3), device=device) * 2 - 1
        
    #     # Measure time for KNN
    #     # torch.cuda.empty_cache()
    #     start_time = time.time()
    #     knn_neighbor(input_xyz, query_xyz)
    #     knn_time = time.time() - start_time
    #     knn_times.append((input_num, query_num_fixed, knn_time))
        
    #     # torch.cuda.empty_cache()
    #     # Measure time for Serial Neighbor
    #     start_time = time.time()
    #     serial_neighbor(input_xyz, query_xyz)
    #     serial_time = time.time() - start_time
    #     serial_times.append((input_num, query_num_fixed, serial_time))
    
    # # Printing results in table format
    # print("Results for varying query points with fixed input points:")
    # print(f"{'Input Points':<15}{'Query Points':<15}{'KNN Time (s)':<15}{'Serial Time (s)':<15}")
    # print("-" * 60)
    # for (input_num, query_num, knn_time), (_, _, serial_time) in zip(knn_times[:len(query_num_range)], serial_times[:len(query_num_range)]):
    #     print(f"{input_num:<15}{query_num:<15}{knn_time:<15.6f}{serial_time:<15.6f}")
    
    # print("\nResults for varying input points with fixed query points:")
    # print(f"{'Input Points':<15}{'Query Points':<15}{'KNN Time (s)':<15}{'Serial Time (s)':<15}")
    # print("-" * 60)
    # for (input_num, query_num, knn_time), (_, _, serial_time) in zip(knn_times[len(query_num_range):], serial_times[len(query_num_range):]):
    #     print(f"{input_num:<15}{query_num:<15}{knn_time:<15.6f}{serial_time:<15.6f}")

if __name__ == "__main__":
    main()
