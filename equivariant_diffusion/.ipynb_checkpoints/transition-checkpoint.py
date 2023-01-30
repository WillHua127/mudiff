import torch


class DiscreteUniformTransition:
    def __init__(self, e_classes: int):
        self.E_classes = e_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)

        return q_e

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qe (bs, de, de)
        """
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e

        return q_e


class MarginalUniformTransition:
    def __init__(self, e_marginals):
        self.E_classes = len(e_marginals)
        self.e_marginals = e_marginals

        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)

    def get_Qt(self, beta_t, device):
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)

        return q_e

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qe (bs, de, de)
        """
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_e = self.u_e.to(device)

        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e

        return q_e