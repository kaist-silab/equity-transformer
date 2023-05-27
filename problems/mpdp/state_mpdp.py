import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class State_MPDP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc

    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    agent_idx: torch.Tensor # order of agents are currently moving
    prev_a: torch.Tensor # Previous actions
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor # Keeps track of tour lengths corresponding to each agent
    cur_coord: torch.Tensor # Keeps track of current coordinates
    i: torch.Tensor  # Keeps track of step
    count_depot: torch.Tensor # Number of depot visited
    left_request : torch.Tensor # Number of left cities
    depot_distance : torch.Tensor  # Distance from depot to all cities
    remain_pickup_max_distance : torch.Tensor # Max distance from depot among left pickup nodes
    remain_delivery_max_distance : torch.Tensor # Max distance from depot among left delivery nodes
    remain_sum_paired_distance : torch.Tensor # Sum of distance left paired pickup and delivery nodes
    to_delivery: torch.Tensor # Keep track of wether the agent can go delivery nodes
    add_pd_distance : torch.Tensor # Distance from corresponding pickup to delivery
    longest_lengths : torch.Tensor #Worst case length of each agent
     
    
    
    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            
              # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
#                 used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                longest_lengths=self.longest_lengths[key],
                cur_coord=self.cur_coord[key],
                to_delivery=self.to_devliery[key],
                count_depot = self.count_depot[key],
                agent_idx=self.agent_idx[key],
                left_request = self.left_request[key],
                depot_distance = self.depot_distance[key],
                remain_pickup_max_distance = self.remain_pickup_max_distance[key],
                remain_delivery_max_distance = self.remain_delivery_max_distance[key],
                remain_sum_paired_distance = self.remain_sum_paired_distance[key],
                add_pd_distance = self.add_pd_distance[key]
            )

    @staticmethod
    def initialize(input, agent_num, visited_dtype=torch.uint8):
        
        depot = input['depot']
        depot = depot.repeat(1, agent_num + 1, 1)
        loc = input['loc']
        left_request = loc.size(1) // 2
        whole_instance = torch.cat((depot, loc), dim=1)
        # Distance from all nodes between each other
        distance = torch.cdist(whole_instance, whole_instance, p=2)
        
        # Distance paires between pickup and delivery nodes
        pickup_delivery_distance = distance[:,agent_num+1:agent_num+1+left_request,agent_num+1:]
        
        index = torch.arange(left_request, 2*left_request, device=depot.device)[None,:,None]
        index = index.repeat(pickup_delivery_distance.shape[0], 1, 1)
        add_pd_distance = pickup_delivery_distance.gather(-1,index)
        add_pd_distance = add_pd_distance.squeeze(-1)


        remain_pickup_max_distance = distance[:,0,:agent_num + 1 + left_request].max(dim=-1, keepdim=True)[0]
        remain_delivery_max_distance = distance[:,0,agent_num + 1 + left_request:].max(dim=-1, keepdim=True)[0]
        remain_sum_paired_distance = add_pd_distance.sum(dim=-1, keepdim=True)
        
        # Distance from depot to all nodes
        # Delivery nodes should consider the sum of distance from depot to paired pickup nodes and pickup nodes to delivery nodes
        distance[:,0,agent_num+1: agent_num+1 + left_request] = distance[:,0,agent_num+1: agent_num+1 + left_request] + distance[:,0,agent_num+1 + left_request:]
        
        # Distance from depot to all nodes
        depot_distance = distance[:,0,:]
        depot_distance[:,agent_num+1:agent_num+1 + left_request] = depot_distance[:,agent_num+1:agent_num+1 + left_request]#+ add_pd_distance

        batch_size, n_loc, _ = loc.size()
        to_delivery=torch.cat([torch.ones(batch_size, 1, n_loc // 2 + agent_num + 1, dtype=torch.uint8, device=loc.device),
            torch.zeros(batch_size, 1, n_loc // 2, dtype=torch.uint8, device=loc.device)], dim=-1)  # [batch_size, 1, graph_size+1], [1,1...1, 0...0]
        if len(depot.size()) == 2:
            return State_MPDP(
            coords=torch.cat((depot[:, None, :], loc), -2),
#             demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
#             used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, agent_num, device=loc.device),
            longest_lengths=torch.zeros(batch_size, agent_num, device=loc.device),
            cur_coord=input['depot'][:, None, :],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            to_delivery=to_delivery,
            count_depot = torch.zeros(batch_size, 1, dtype=torch.int64, device=loc.device),
            agent_idx=torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
            left_request = left_request * torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
            remain_pickup_max_distance= remain_pickup_max_distance,
            remain_delivery_max_distance = remain_delivery_max_distance,
            depot_distance = depot_distance,
            remain_sum_paired_distance = remain_sum_paired_distance,
            add_pd_distance = add_pd_distance
        )
        else:
            return State_MPDP(
            coords=torch.cat((depot, loc), -2),
#             demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
#             used_capacity=demand.new_zeros(batch_size, 1),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc + agent_num + 1,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, agent_num, device=loc.device),
            longest_lengths=torch.zeros(batch_size, agent_num, device=loc.device),
            cur_coord=input['depot'],  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Vector with length num_steps
            to_delivery=to_delivery,
            count_depot = torch.zeros(batch_size, 1, dtype=torch.int64, device=loc.device),
            agent_idx=torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
            left_request = left_request * torch.ones(batch_size, 1, dtype=torch.long, device=loc.device),
            remain_pickup_max_distance= remain_pickup_max_distance,
            remain_delivery_max_distance = remain_delivery_max_distance,
            depot_distance = depot_distance,
            remain_sum_paired_distance = remain_sum_paired_distance,
            add_pd_distance = add_pd_distance
        )

       

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + (self.coords[self.ids, 0, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"
        agent_num = self.lengths.size(1)
        # Update the state
        n_loc = self.to_delivery.size(-1) - agent_num - 1  # number of customers
        
        new_to_delivery = (selected + n_loc // 2) % (n_loc + agent_num + 1)  # the pair node of selected node
        new_to_delivery = new_to_delivery[:, None]
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected

        # if selected node is pickup, is_request = True
        is_request = (prev_a > agent_num) & (prev_a <= agent_num + n_loc // 2)
        
        self.left_request[is_request] -= 1
        depot_distance = self.depot_distance.scatter(-1, prev_a, 0)


        add_pd = self.add_pd_distance[is_request.squeeze(-1),:].gather(-1, prev_a[is_request.squeeze(-1),:] - agent_num - 1)
        self.longest_lengths[is_request.squeeze(-1),:] = self.longest_lengths[is_request.squeeze(-1),:].scatter_add(-1, self.count_depot[is_request.squeeze(-1),:], add_pd)
        self.add_pd_distance[is_request.squeeze(-1),:] = self.add_pd_distance[is_request.squeeze(-1),:].scatter(-1, prev_a[is_request.squeeze(-1),:] - agent_num - 1, 0)
        remain_sum_paired_distance = self.add_pd_distance.sum(-1, keepdim=True)
        remain_pickup_max_distance = depot_distance[:,:agent_num + 1 + n_loc//2].max(dim=-1, keepdim=True)[0]
        remain_delivery_max_distance = depot_distance[:,agent_num + 1 + n_loc//2:].max(dim=-1, keepdim=True)[0]

        cur_coord = self.coords[self.ids, selected]
        #To calculate makespan
        path_lengths = (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)
        lengths = self.lengths.scatter_add(-1, self.count_depot, path_lengths)

        # if visit depot then plus one to count_depot
        self.count_depot[(selected == self.agent_idx) & (self.agent_idx < agent_num)] += torch.ones(self.count_depot[(selected == self.agent_idx) & (self.agent_idx < agent_num)].shape, dtype=torch.int64, device=self.count_depot.device)
        
        # agent_idx is added by 1 if the current agent comes back to depot
        agent_idx = (self.count_depot + 1) * torch.ones(selected.size(0), 1, dtype=torch.long, device=self.count_depot.device)
        
        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
            to_delivery = self.to_delivery.scatter(-1, new_to_delivery[:, :, None], 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = mask_long_scatter(self.visited_, prev_a - 1)

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1, to_delivery=to_delivery, agent_idx=agent_idx,
            depot_distance=depot_distance, remain_pickup_max_distance=remain_pickup_max_distance,remain_delivery_max_distance=remain_delivery_max_distance,
            remain_sum_paired_distance = remain_sum_paired_distance
        )

    def all_finished(self):
        return self.visited.all()


    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """
        # if self.i.item() != 0:
        #     self.visited_[:,:,0] = 0

        visited_loc = self.visited_.clone()
        
        
        agent_num = self.lengths.size(1)
        n_loc = visited_loc.size(-1) - agent_num - 1  # num of customers 
        batch_size = visited_loc.size(0)
        agent_idx = self.agent_idx[:,None,:]
        mask_loc = visited_loc.to(self.to_delivery.device) | (1 - self.to_delivery)
        
        #depot
        if self.i.item() != 0:
            mask_loc[:,:,:agent_num+1] = 1
            
        # if deliver nodes which is assigned agent is complete, then agent can go to depot
            no_item_to_delivery = (visited_loc[:, :, n_loc//2+agent_num+1:] == self.to_delivery[:, :, n_loc//2+agent_num+1:]).all(dim=-1)
            mask_loc[no_item_to_delivery.squeeze(-1), :, :] = mask_loc[no_item_to_delivery.squeeze(-1), :, :].scatter_(-1, agent_idx[no_item_to_delivery.squeeze(-1), :, :], 0)
            
            condition = ((self.count_depot == agent_num-1) & ((visited_loc[:,:,agent_num+1:] == 0).sum(dim=-1)!= 0))
            
            mask_loc[...,agent_num][condition] = 1

        if self.i == 0:  # [0,1...1]
            return torch.cat([torch.zeros(batch_size, 1, 1, dtype=torch.uint8, device=mask_loc.device),
                            torch.ones(batch_size, 1, n_loc + agent_num, dtype=torch.uint8, device=mask_loc.device)], dim=-1) > 0
        return mask_loc > 0  # return true/false

    def construct_solutions(self, actions):
        return actions
