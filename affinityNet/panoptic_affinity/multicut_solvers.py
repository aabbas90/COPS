import torch
from lpmp_py.raw_solvers import amwc_solver, mwc_solver
import numpy as np 
import torch.multiprocessing as mp

def get_edge_indices(image_shape, edge_distances, edge_sampling_intervals):
    indices = np.arange(np.prod(image_shape)).reshape(image_shape).astype(np.int32)
    edge_indices = {}
    current_si = 1
    for (i, e_d) in enumerate(edge_distances):
        if edge_sampling_intervals is not None:
            current_si = edge_sampling_intervals[i]

        left_offset = int(e_d / 2.0)
        right_offset = left_offset
        if e_d == 1:
            left_offset = 0
            right_offset = 1

        valid_left_offset = int(np.ceil(left_offset / current_si))
        valid_right_offset = int(np.ceil(right_offset / current_si))
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None
        e1_row = np.meshgrid(np.arange(-left_offset, image_shape[0] - left_offset, current_si)[valid_left_offset:output_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')
        e1_row = np.ravel_multi_index(e1_row, dims = image_shape)

        e2_row = np.meshgrid(np.arange(right_offset, image_shape[0] + right_offset, current_si)[valid_left_offset:output_right_offset], 
                            np.arange(0, image_shape[1], current_si), indexing='ij')

        e2_row = np.ravel_multi_index(e2_row, dims = image_shape)

        edge_indices[str(e_d) + 'row'] = {'e1': e1_row, 'e2': e2_row, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

        e1_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(-left_offset, image_shape[1] - left_offset, current_si)[valid_left_offset:output_right_offset], indexing='ij')

        e1_col = np.ravel_multi_index(e1_col, dims = image_shape)

        e2_col = np.meshgrid(np.arange(0, image_shape[0], current_si),
                            np.arange(right_offset, image_shape[1] + right_offset, current_si)[valid_left_offset:output_right_offset], indexing='ij')
        e2_col = np.ravel_multi_index(e2_col, dims = image_shape)

        edge_indices[str(e_d) + 'col'] = {'e1': e1_col, 'e2': e2_col, 'valid_left_offset': valid_left_offset, 'valid_right_offset': valid_right_offset}

    return edge_indices

def get_edge_list(edge_costs, edge_indices, edge_distances):
    edge_costs_1d = []
    edge_indices_1 = []
    edge_indices_2 = []

    for (i, e_d) in enumerate(edge_distances):
        e1_row = edge_indices[str(e_d) + 'row']['e1']
        e2_row = edge_indices[str(e_d) + 'row']['e2']
        valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_row_costs = edge_costs[i][0, valid_left_offset:output_right_offset, :]
        edge_costs_1d.append(current_row_costs.flatten())
        edge_indices_1.append(e1_row.flatten())
        edge_indices_2.append(e2_row.flatten())

        e1_col = edge_indices[str(e_d) + 'col']['e1']
        e2_col = edge_indices[str(e_d) + 'col']['e2']
        valid_left_offset = edge_indices[str(e_d) + 'col']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'col']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_col_costs = edge_costs[i][1, :, valid_left_offset:output_right_offset]
        edge_costs_1d.append(current_col_costs.flatten())
        edge_indices_1.append(e1_col.flatten())
        edge_indices_2.append(e2_col.flatten())
    edge_costs_cpu_1d = np.concatenate(edge_costs_1d, 0)
    edge_indices_1 = np.concatenate(edge_indices_1, 0)
    edge_indices_2 = np.concatenate(edge_indices_2, 0)

    return np.stack((edge_indices_1, edge_indices_2), 1), edge_costs_cpu_1d

def get_edge_labels_from_1d(edge_costs, edge_labels_1d, edge_indices, edge_distances, dtype = np.uint8):
    edge_labels = []
    start_index = 0
    for (i, e_d) in enumerate(edge_distances):
        valid_left_offset = edge_indices[str(e_d) + 'row']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'row']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        current_costs_shape = edge_costs[i].shape
        full_edge_labels = np.zeros(current_costs_shape, dtype=dtype)
        valid_edge_labels = np.zeros((current_costs_shape[1] - valid_left_offset - valid_right_offset, current_costs_shape[2]), dtype=dtype)
        valid_shape = valid_edge_labels.shape
        valid_edge_labels = valid_edge_labels.flatten() 
        valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
        full_edge_labels[0, valid_left_offset:output_right_offset, :] = valid_edge_labels.reshape(valid_shape)

        start_index += valid_edge_labels.size

        valid_left_offset = edge_indices[str(e_d) + 'col']['valid_left_offset']
        valid_right_offset = edge_indices[str(e_d) + 'col']['valid_right_offset']
        output_right_offset = -valid_right_offset if valid_right_offset > 0 else None

        valid_edge_labels = np.zeros((current_costs_shape[1], current_costs_shape[2] - valid_left_offset - valid_right_offset), dtype=dtype)
        valid_shape = valid_edge_labels.shape
        valid_edge_labels = valid_edge_labels.flatten() 
        valid_edge_labels = edge_labels_1d[start_index:start_index + valid_edge_labels.size] 
        full_edge_labels[1, :, valid_left_offset:output_right_offset] = valid_edge_labels.reshape(valid_shape)
        start_index += valid_edge_labels.size
        edge_labels.append(full_edge_labels)
    return edge_labels


def solve_mc_grad_avg(batch_index, orig_semantic_labels_cpu, orig_edge_labels_cpu, node_costs_cpu, grad_node_costs_cpu, edge_costs_cpu, index_to_category, edge_indices, edge_distances, edge_sampling_intervals, min_pert, max_pert, num_samples, num_semantic_classes, return_dict):
    edge_indices_1d, edge_costs_1d = get_edge_list(edge_costs_cpu, edge_indices, edge_distances)
    _, orig_edge_labels_1d = get_edge_list(orig_edge_labels_cpu, edge_indices, edge_distances)
    num_classes = node_costs_cpu.shape[0]
    node_labels_img_shape = node_costs_cpu.shape[1:]
    node_labels_num_pixels = np.prod(node_labels_img_shape)
    node_costs_cpu = node_costs_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_classes)
    grad_node_costs_cpu = grad_node_costs_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_classes)
    orig_semantic_labels_cpu = orig_semantic_labels_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_semantic_classes)
    grad_semantic_avg = None
    grad_edge_avg = None 
    for s in range(num_samples):
        pert = np.random.uniform(low = min_pert, high=max_pert)
        current_pert_node_costs = node_costs_cpu + pert * grad_node_costs_cpu
        pert_instance_labels, pert_edge_labels, _ = mwc_solver(current_pert_node_costs, edge_indices_1d, edge_costs_1d)
        pert_semantic_labels = get_semantic_labels_from_instance_labels(pert_instance_labels, index_to_category, num_semantic_classes)
        if s == 0:
            grad_semantic_avg = (pert_semantic_labels - orig_semantic_labels_cpu) / pert
            grad_edge_avg = (pert_edge_labels - orig_edge_labels_1d) / pert
        else:
            grad_semantic_avg += (pert_semantic_labels - orig_semantic_labels_cpu) / pert
            grad_edge_avg += (pert_edge_labels - orig_edge_labels_1d) / pert
    grad_semantic_avg = grad_semantic_avg / num_samples
    grad_edge_avg = grad_edge_avg / num_samples
    grad_semantic_avg = grad_semantic_avg.transpose().reshape((num_semantic_classes, node_labels_img_shape[0], node_labels_img_shape[1]))
    grad_edge_avg = get_edge_labels_from_1d(edge_costs_cpu, grad_edge_avg, edge_indices, edge_distances, np.float32)

    return_dict[batch_index] = (grad_semantic_avg, grad_edge_avg)

def get_semantic_labels_from_instance_labels(instance_labels, index_to_category, num_classes):
    semantic_labels = np.zeros((instance_labels.shape[0], num_classes), dtype=np.float32)
    for idx, cat in index_to_category.items():
        semantic_labels[:, cat] += instance_labels[:, idx]
    return semantic_labels

def solve_amc(batch_index, node_costs_cpu, edge_costs_cpu, edge_indices, edge_distances, edge_sampling_intervals, thing_ids, ignore_instance_size_amc, return_dict):
    edge_indices_1d, edge_costs_1d = get_edge_list(edge_costs_cpu, edge_indices, edge_distances)
    num_classes = node_costs_cpu.shape[0]
    node_labels_img_shape = node_costs_cpu.shape[1:]
    node_labels_num_pixels = np.prod(node_labels_img_shape)
    node_costs_cpu = node_costs_cpu.transpose(1, 2, 0).reshape(node_labels_num_pixels, num_classes)
    partitionable = np.array([False] * num_classes, dtype='bool')
    for t in thing_ids:
        partitionable[t] = True

    node_labels, node_instance_ids, edge_labels_1d, solver_cost = amwc_solver(node_costs_cpu, edge_indices_1d, edge_costs_1d, partitionable)
    node_instance_ids = node_instance_ids.reshape(node_labels_img_shape)

    node_labels = node_labels.transpose().reshape((num_classes, node_labels_img_shape[0], node_labels_img_shape[1]))
    edge_labels = get_edge_labels_from_1d(edge_costs_cpu, edge_labels_1d, edge_indices, edge_distances)
    panoptic_ids_one_hot = []
    index_to_class_labels = []
    for class_id in range(num_classes):
        current_class_mask = node_labels[class_id, ::] == 1
        if class_id not in thing_ids:
            if np.any(current_class_mask):
                panoptic_ids_one_hot.append(current_class_mask)
                index_to_class_labels.append(class_id)
        else:
            instance_ids_within_class, counts = np.unique(node_instance_ids[current_class_mask], return_counts=True)
            for instance_id, cnt in zip(instance_ids_within_class, counts):
                if cnt < ignore_instance_size_amc:
                    continue
                current_mask = node_instance_ids == instance_id
                panoptic_ids_one_hot.append(current_mask)
                index_to_class_labels.append(class_id)
    panoptic_ids_one_hot = np.stack(panoptic_ids_one_hot)
    index_to_class_labels = np.stack(index_to_class_labels)

    return_dict[batch_index] = (node_labels, edge_labels, panoptic_ids_one_hot, index_to_class_labels, solver_cost)

def solve_mc_grad_avg_batch(semantic_labels_batch, edge_labels_batch, node_costs_batch, grad_node_costs_batch, edge_costs_batch, edge_indices, edge_distances, edge_sampling_intervals, all_index_to_category, min_pert, max_pert, num_samples, num_semantic_classes):
    if isinstance(node_costs_batch, torch.Tensor):
        node_costs_batch = torch.unbind(node_costs_batch, dim=0)
    if isinstance(grad_node_costs_batch, torch.Tensor):
        grad_node_costs_batch = torch.unbind(grad_node_costs_batch, dim=0)
    if isinstance(semantic_labels_batch, torch.Tensor):
        semantic_labels_batch = torch.unbind(semantic_labels_batch, dim=0)

    batch_size = len(node_costs_batch)
    node_costs_batch_cpu = [n.detach().cpu().numpy() for n in node_costs_batch]
    grad_node_costs_batch_cpu = [n.detach().cpu().numpy() for n in grad_node_costs_batch]
    semantic_labels_batch_cpu = [s.detach().cpu().numpy() for s in semantic_labels_batch]
    edge_costs_batch_cpu = []
    edge_labels_batch_cpu = []
    for b in range(batch_size):
        current_batch_edge_costs = []
        current_batch_edge_labels = []
        for i in range(len(edge_costs_batch)): # Iterate over different edge distances.
            current_batch_edge_costs.append(edge_costs_batch[i][b, ::].detach().cpu().numpy())
            current_batch_edge_labels.append(edge_labels_batch[i][b, ::].detach().cpu().numpy())
        edge_costs_batch_cpu.append(current_batch_edge_costs)
        edge_labels_batch_cpu.append(current_batch_edge_labels)

    if batch_size == 1:
        b = 0
        return_dict = {}
        solve_mc_grad_avg(b, semantic_labels_batch_cpu[b], edge_labels_batch_cpu[b], node_costs_batch_cpu[b], grad_node_costs_batch_cpu[b], edge_costs_batch_cpu[b], all_index_to_category[b], edge_indices, edge_distances, edge_sampling_intervals, min_pert, max_pert, num_samples, num_semantic_classes, return_dict)
    else:
        ctx = mp.get_context('fork')
        manager = ctx.Manager()
        return_dict = manager.dict()
        workers = []
        for b in range(batch_size): 
            worker = ctx.Process(target=solve_mc_grad_avg, args=(b, semantic_labels_batch_cpu[b], edge_labels_batch_cpu[b], node_costs_batch_cpu[b], grad_node_costs_batch_cpu[b], edge_costs_batch_cpu[b], all_index_to_category[b], edge_indices, edge_distances, edge_sampling_intervals, min_pert, max_pert, num_samples, num_semantic_classes, return_dict))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                import sys
                print(f"ERROR: There was an error during multiprocessing with error code: {worker.exitcode}, in worker: {worker.name}. Possibly too many parallel tasks!")
                sys.exit(0)

    grad_node_labels_batch = []
    grad_edge_labels_batch = []  
    for b in sorted(return_dict.keys()):
        g_node_labels, g_edge_labels = return_dict[b]
        grad_node_labels_batch.append(torch.from_numpy(g_node_labels))
        grad_edge_labels_batch.append(g_edge_labels)

    grad_edge_labels_batch_reorg = []
    for i in range(len(grad_edge_labels_batch[0])): # Iterate over different edge distances.
        current_distance_g_edge_labels = []
        for b in range(batch_size):
            current_distance_g_edge_labels.append(grad_edge_labels_batch[b][i])
        
        grad_edge_labels_batch_reorg.append(torch.from_numpy(np.stack(current_distance_g_edge_labels, 0))) #.to(torch.float32).to(device))

    return grad_node_labels_batch, grad_edge_labels_batch_reorg

def solve_amc_batch(node_costs_batch, edge_costs_batch, edge_indices, edge_distances, edge_sampling_intervals, ignore_instance_size_amc, thing_ids = None):
    panoptic_ids_one_hot = None
    if isinstance(node_costs_batch, torch.Tensor):
        node_costs_batch = torch.unbind(node_costs_batch, dim=0)

    batch_size = len(node_costs_batch)
    node_costs_batch_cpu = [n.detach().cpu().numpy() for n in node_costs_batch]
    edge_costs_batch_cpu = []
    for b in range(batch_size):
        current_batch_edge_costs = []
        for i in range(len(edge_costs_batch)): # Iterate over different edge distances.
            current_batch_edge_costs.append(edge_costs_batch[i][b, ::].detach().cpu().numpy())
        edge_costs_batch_cpu.append(current_batch_edge_costs)

    if batch_size == 1:
        b = 0
        return_dict = {}
        solve_amc(b, node_costs_batch_cpu[b], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, thing_ids, ignore_instance_size_amc, return_dict)
    else:
        ctx = mp.get_context('fork')
        manager = ctx.Manager()
        return_dict = manager.dict()
        workers = []
        for b in range(batch_size): 
            worker = ctx.Process(target=solve_amc, args=(b, node_costs_batch_cpu[b], edge_costs_batch_cpu[b], edge_indices, edge_distances, edge_sampling_intervals, thing_ids, ignore_instance_size_amc, return_dict))
            workers.append(worker)
        [w.start() for w in workers]  
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                import sys
                print(f"ERROR: There was an error during multiprocessing with error code: {worker.exitcode}, in worker: {worker.name}. Possibly too many parallel tasks!")
                sys.exit(0)

    node_labels_batch = []
    edge_labels_batch = []  
    panoptic_ids_one_hot_batch = []
    index_to_class_labels_batch = []

    solver_costs = []
    for b in sorted(return_dict.keys()):
        node_labels, current_edge_labels, panoptic_ids_one_hot, index_to_class_labels, solver_cost = return_dict[b]
        node_labels_batch.append(torch.from_numpy(node_labels))
        edge_labels_batch.append(current_edge_labels)
        solver_costs.append(solver_cost)
        assert(len(current_edge_labels) == len(edge_distances))
        panoptic_ids_one_hot_batch.append(torch.from_numpy(panoptic_ids_one_hot))
        index_to_class_labels_batch.append(torch.from_numpy(index_to_class_labels))

    edge_labels_batch_reorg = []
    for i in range(len(edge_labels_batch[0])): # Iterate over different edge distances.
        current_distance_edge_labels = []
        for b in range(batch_size):
            current_distance_edge_labels.append(edge_labels_batch[b][i])
        
        edge_labels_batch_reorg.append(torch.from_numpy(np.stack(current_distance_edge_labels, 0)))

    return node_labels_batch, tuple(edge_labels_batch_reorg), tuple(panoptic_ids_one_hot_batch), tuple(index_to_class_labels_batch), tuple(solver_costs)

class AsymmetricMultiCutSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs): #ctx, edge_indices, params, node_costs, edge_costs):
        ctx.set_materialize_grads(False)
        edge_indices = inputs[0]
        params = inputs[1]
        node_costs = inputs[2]
        edge_costs = inputs[3:3+len(params['edge_distances'])]

        node_labels, edge_labels, panoptic_ids_one_hot, index_to_class_labels, solver_costs = solve_amc_batch(node_costs, edge_costs, edge_indices, params['edge_distances'], params['edge_sampling_intervals'], params['ignore_instance_size_amc'], params['thing_ids'])
        node_labels = torch.stack(node_labels, 0)
        ctx.params = params
        ctx.device = node_costs.device
        ctx.edge_indices = edge_indices

        edge_labels = tuple([e.to(torch.float32) for e in edge_labels])

        panoptic_ids_one_hot = tuple([p.to(torch.float32) for p in panoptic_ids_one_hot])
        [ctx.mark_non_differentiable(p) for p in panoptic_ids_one_hot] 
        
        index_to_class_labels = tuple([ind for ind in index_to_class_labels]) 
        [ctx.mark_non_differentiable(ind) for ind in index_to_class_labels]

        out = (node_labels.to(torch.float32), ) + edge_labels + panoptic_ids_one_hot + index_to_class_labels
        return out

    @staticmethod
    def backward(*grad_inputs):
        """
        Backward pass computation.

        @param ctx: context from the forward pass
        @param grad_node_labels: "dL / d node_labels"
        @param grad_edge_labels: "dL / d edge_labels" 
        @param grad_panoptic_ids_one_hot: Contains "dL / d panoptic_ids_one_hot" 
            does not contain meaningful value as gradients are computed in MWC. 
        @param grad_index_to_class_labels: Contains "dL / d index_to_class_labels" 
            does not contain meaningful value as non-differentiable and not required to be. 
        @return: gradient dL / node_costs, dL / edge_costs
        """
        ctx = grad_inputs[0]
        params = ctx.params
        num_edge_arrays = len(params['edge_distances'])
        grad_node_costs = grad_inputs[1]
        grad_edge_costs = [g for g in grad_inputs[2:2 + num_edge_arrays]]        
        
        # Incoming gradients are already correct (computed through MWC)
        grad_node_costs = grad_node_costs.to(ctx.device)
        grad_edge_costs = [g.to(ctx.device) for g in grad_edge_costs]
        out = (None, ) + (None, ) + (grad_node_costs, ) + tuple(grad_edge_costs)
        return out

class AsymmetricMulticutModule(torch.nn.Module):
    """
    Torch module for handling batches of Asymmetric Multicut Instances
    """
    def __init__(self, thing_ids = None):
        """
        @param thing_ids: List of indices of segmentation targets which correspond to thing classes.
        """
        super().__init__()
        self.solver = AsymmetricMultiCutSolver()
        self.image_size = None
        self.params = {"thing_ids": thing_ids}

    def forward(self, node_costs_batch, edge_costs_batch, edge_distances, edge_sampling_intervals, ignore_instance_size_amc):
        """
        """
        self.params.update({"edge_distances": edge_distances,
                            "edge_sampling_intervals": edge_sampling_intervals, 
                            "ignore_instance_size_amc": ignore_instance_size_amc})
        # Update edge indices if the image size is changed:
        if self.image_size is None or node_costs_batch[0, 0].shape != self.image_size:
            self.image_size = node_costs_batch[0, 0].shape
            self.edge_indices = get_edge_indices(self.image_size, edge_distances, edge_sampling_intervals)
            self.params['edge_indices'] = self.edge_indices
        model_input = (self.edge_indices, ) + (self.params, ) + (node_costs_batch, ) + tuple(edge_costs_batch) 
        out = self.solver.apply(*model_input)
        return out # Tuple
