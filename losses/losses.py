from torch.nn import Module
from scipy.stats import gaussian_kde

class UnbalancedSELoss(Module):
    """
    Loss function for unbalanced regression task which multiplies
    the inverse of Kernel Density Estimation (KDE) entrywise with
    the squared loss, before taking the mean.
    See https://towardsdatascience.com/strategies-and-tactics-for-regression-on-imbalanced-data-61eeb0921fca
    for a more detailed description.
    Warning: this loss function depends on the dataset!
    """

    def __init__(self, dataset, epsilon=1e-9):
        super(UnbalancedSELoss, self).__init__()
        self.kde, self.inverse_kde_mean = self.get_kde(dataset)
        self.epsilon = epsilon

    def forward(self, x, y):
        x_pdf = self.kde(x.detach().numpy()).clip(self.epsilon, None) #y plugged into kde
        x_pdf = torch.tensor(x_pdf) * self.inverse_kde_mean
        error_vector = (x - y) ** 2 / x_pdf
        return error_vector.mean()

    @staticmethod
    def get_kde(dataset):
        """
        Computes the KDE based on the dataset provided
        using the scipy stats package
        """
        ys = np.zeros(len(dataset))
        for i in range(len(dataset)):
            ys[i] = dataset[i]["y"].item()
        kde = gaussian_kde(ys)
        inverse_kde_mean = (1 / kde(ys)).mean()
        return kde, inverse_kde_mean