




class CostmapDataset(Dataset):

    def __init__(self, n_samples = 1000000 H=64, W=64):
        self.H = H
        self.W = W
        self.cost = np.zeros((H, W), dtype=np.float32)
        self.obstacles = []
        self.obstacles_byclass = {}
        self.robot = [H, W]
        self.goal = [10, 10]
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, obstacles:dict, goal: np.ndarray):
        
        cm = Costmap()

        cm.goal = goal
        costmaps, occupancy_maps = cm.calculateCost(obstacles, goal)

        #Build conditioning vectors

        #BINARY occupancy map
        #goal state
        #radius


        return , x0
