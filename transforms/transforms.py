from torch_geometrc.nn import BaseTransform


class Normalize(BaseTransform):
    "Normalizes Data by dividing each data column by max entry"

    def __init__(self):
        pass

    def __call__(self, data):
        x = data.x
        max_x = x.max(dim=0).values

        #Replacing zero entries by 1
        max_x = torch.where(max_x != 0, max_x, torch.ones_like(max_x))
        data.x = x / max_x

        #Replacing zero entries by 1
        y = data.y
        max_y = y.max(dim=0).values
        max_y = torch.where(max_y != 0, max_y, torch.ones_like(max_y))
        data.y = y / max_y

        return data


class ToHetero(BaseTransform):
    """
    Transforms torch geometric Data to Heterogenous Data
    by viewing the second node feature as node type
    """

    def __init__(self):
        pass

    def __call__(self, data):
        data.x = data.x[:,0]
        node_types = data.x[:,1].to(int)

        return data.to_heterogenous(node_type=node_types, node_type_names=["standard", "slack"])