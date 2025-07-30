class Config(object):
    def __init__(self):
        self.device = 'cpu'
        self.epoches = 30
        self.seed = 12
        self.learning_rate = 3e-5
        # self.Td_x_hidden_dim=[2000,1024,2000]
        # self.Td_y_hidden_dim=[2000,1024,2000]
        # self.Td_z_hidden_dim=[2000,1024,3500]
        # self.Yita_one_hidden_dim=[2000,1024,2000]
        # self.Yita_two_hidden_dim=[2000,1024,2000]
        # self.Yita_three_hidden_dim=[2000,1024,2000]
        self.Td_x_hidden_dim = [3500, 2048, 3500]
        self.Td_y_hidden_dim = [3500, 2048, 3500]
        self.Td_z_hidden_dim = [3500, 2048, 3500]
        self.Yita_one_hidden_dim = [3500, 2048, 3500]
        self.Yita_two_hidden_dim = [3500, 2048, 3500]
        self.Yita_three_hidden_dim = [3500, 2048, 3500]

        # self.Td_x_hidden_dim = [1500, 512, 1500]
        # self.Td_y_hidden_dim = [1500, 512, 1500]
        # self.Td_z_hidden_dim = [1500, 512, 1500]
        # self.Yita_one_hidden_dim = [1500, 512, 1500]
        # self.Yita_two_hidden_dim = [1500, 512, 1500]
        # self.Yita_three_hidden_dim = [1500, 512, 1500]


