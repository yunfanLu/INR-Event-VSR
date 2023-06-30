import numpy as np
import torch
import torch.nn as nn


class CNNDecoder(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        assert hidden_layers >= 2, "hidden_layers should be >= 2"
        self.net = []

        self.net.append(nn.Conv2d(in_features, hidden_features, 1))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers - 2):
            self.net.append(nn.Conv2d(hidden_features, hidden_features, 1))
            self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(hidden_features, out_features, 1))
        self.net.append(nn.ReLU())
        self.net = nn.Sequential(*self.net)

        self.in_features, self.hidden_features = in_features, hidden_features

    def forward(self, input):
        output = self.net(input)
        return output


class MLPDecoder(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        assert hidden_layers >= 2, "hidden_layers should be >= 2"
        self.net = []
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(nn.ReLU())
        for i in range(hidden_layers - 2):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net.append(nn.ReLU())
        # self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

        self.in_features, self.hidden_features = in_features, hidden_features

    def forward(self, input):
        # input: (B, C, H, W)
        # output: (B, out_features, H, W)
        B, C, H, W = input.shape
        h = input.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        h = self.net(h)  # (B*H*W, out_features)
        output = h.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_features, H, W)
        return output


class SirenDecoder(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            # self.net.append(final_linear)
            self.last_linear = final_linear
        else:
            # self.net.append(SineLayer(hidden_features, out_features,
            #                           is_first=False, omega_0=hidden_omega_0))
            self.last_linear = SineLayer(
                hidden_features,
                out_features,
                is_first=False,
                omega_0=hidden_omega_0,
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # input: (B, C, H, W)
        # output: (B, out_features, H, W)
        B, C, H, W = coords.shape
        h = coords.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)
        # h = h.clone().detach().requires_grad_(True)# allows to take derivative w.r.t. input
        h = self.net(h)
        h = self.last_linear(h)
        output = h.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, out_features, H, W)

        return output


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1.0 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
