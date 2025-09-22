import pytest

from typeguard import TypeCheckError
import torch
import numpy as np

from clair_torch.common.enums import InterpMode
from clair_torch.models.icrf_model import ICRFModelDirect, ICRFModelPCA


class TestICRFModelPCA:

    n_points, channels, n_components = 256, 3, 5
    pca_basis = torch.ones((channels, n_components, n_points), dtype=torch.float32)
    interpolation_mode = InterpMode.LINEAR
    initial_power = 3.0
    expected_initial_icrf = torch.transpose(
        torch.linspace(0, 1, n_points).unsqueeze(1).repeat(1, channels) ** initial_power, 0, 1)
    expected_x_axis = torch.linspace(0, 1, n_points)

    def test_icrf_model_pca_init_success(self):

        icrf_model = ICRFModelPCA(pca_basis=self.pca_basis, interpolation_mode=self.interpolation_mode,
                                  initial_power=self.initial_power, icrf=None)

        assert icrf_model.initial_power == self.initial_power
        assert icrf_model.channels == self.channels
        assert torch.allclose(icrf_model.icrf, self.expected_initial_icrf)
        assert icrf_model.n_points == self.n_points
        assert icrf_model.interpolation_mode == self.interpolation_mode
        assert torch.allclose(icrf_model.x_axis_datapoints, self.expected_x_axis)
        assert torch.allclose(icrf_model.pca_basis, self.pca_basis)

        assert len(icrf_model.p) == self.channels
        assert len(icrf_model.coefficients) == self.channels

        # One exponent per channel.
        for p in icrf_model.p:
            assert torch.allclose(p, torch.tensor(self.initial_power, dtype=torch.float32))

        # n_components coefficients for each channel.
        for p in icrf_model.coefficients:
            assert p.shape == (self.n_components, )
            assert torch.allclose(p, torch.zeros((self.n_components, ), dtype=torch.float32))

        assert icrf_model._fig is None
        assert icrf_model._axs is None
        assert icrf_model._lines_curve is None
        assert icrf_model._lines_deriv is None

    @pytest.mark.parametrize("pca_basis, interpolation_mode, initial_power, icrf", [
        ("bad", interpolation_mode, initial_power, expected_initial_icrf),
        (pca_basis, "bad", initial_power, expected_initial_icrf),
        (pca_basis, interpolation_mode, "bad", expected_initial_icrf),
        (pca_basis, interpolation_mode, initial_power, "bad")
    ])
    def test_icrf_model_pca_init_invalid_args(self, pca_basis, interpolation_mode, initial_power, icrf):

        with pytest.raises(TypeCheckError):

            _ = ICRFModelPCA(pca_basis=pca_basis, interpolation_mode=interpolation_mode, initial_power=initial_power,
                             icrf=icrf)

    def test_icrf_model_pca_icrf_arg_overrides_pca_arg(self):

        override_channels, override_n_points = 4, 300
        overriding_icrf = torch.ones((override_channels, override_n_points), dtype=torch.float32)

        icrf_model = ICRFModelPCA(pca_basis=self.pca_basis, interpolation_mode=self.interpolation_mode,
                                  initial_power=self.initial_power, icrf=overriding_icrf)

        assert icrf_model.channels == override_channels
        assert icrf_model.n_points == override_n_points

    def test_icrf_model_pca_channel_params(self):

        icrf_model = ICRFModelPCA(pca_basis=self.pca_basis, interpolation_mode=self.interpolation_mode,
                                  initial_power=self.initial_power)

        for c in range(self.channels):
            power, coefficients = icrf_model.channel_params(c)
            assert torch.allclose(power, torch.tensor((self.initial_power, ), dtype=torch.float32))
            assert torch.allclose(coefficients, torch.zeros((self.n_components, ), dtype=torch.float32))


class TestICRFModelDirect:

    n_points, channels = 256, 3
    interpolation_mode = InterpMode.LINEAR
    initial_power = 3
    expected_initial_icrf = torch.transpose(
        torch.linspace(0, 1, n_points).unsqueeze(1).repeat(1, channels) ** initial_power, 0, 1)
    expected_x_axis = torch.linspace(0, 1, n_points)

    def test_icrf_model_direct_init_success(self):

        icrf_model = ICRFModelDirect(n_points=self.n_points, channels=self.channels,
                                     interpolation_mode=self.interpolation_mode, initial_power=self.initial_power)

        assert icrf_model.initial_power == self.initial_power
        assert icrf_model.channels == self.channels
        assert torch.allclose(icrf_model.icrf, self.expected_initial_icrf)
        assert icrf_model.n_points == self.n_points
        assert icrf_model.interpolation_mode == self.interpolation_mode
        assert torch.allclose(icrf_model.x_axis_datapoints, self.expected_x_axis)

        assert icrf_model._fig is None
        assert icrf_model._axs is None
        assert icrf_model._lines_curve is None
        assert icrf_model._lines_deriv is None

        assert len(icrf_model.direct_params) == self.channels
        for p in icrf_model.direct_params:
            assert isinstance(p, torch.nn.Parameter)
            assert p.shape == (self.n_points, )

    def test_icrf_model_direct_init_icrf_arg_override(self):

        override_channels, override_n_points = 4, 300
        overriding_icrf = torch.ones((override_channels, override_n_points), dtype=torch.float32)

        icrf_model = ICRFModelDirect(n_points=self.n_points, channels=self.channels,
                                     interpolation_mode=self.interpolation_mode, initial_power=self.initial_power,
                                     icrf=overriding_icrf)

        assert icrf_model.n_points == override_n_points
        assert icrf_model.channels == override_channels

    @pytest.mark.parametrize("n_points, channels, interpolation_mode, initial_power, icrf", [
        ("bad", channels, interpolation_mode, initial_power, expected_initial_icrf),
        (n_points, "bad", interpolation_mode,  initial_power, expected_initial_icrf),
        (n_points, channels, "bad", initial_power, expected_initial_icrf),
        (n_points, channels, interpolation_mode, "bad", expected_initial_icrf),
        (n_points, channels, interpolation_mode, initial_power, "bad")
    ])
    def test_icrf_model_pca_init_invalid_args(self, n_points, channels, interpolation_mode, initial_power, icrf):

        with pytest.raises(TypeCheckError):
            _ = ICRFModelDirect(n_points=n_points, channels=channels, interpolation_mode=interpolation_mode,
                                initial_power=initial_power, icrf=icrf)

    def test_icrf_model_direct_channel_params(self):

        channelwise_icrfs = [torch.linspace(0, 1, self.n_points) ** c for c in range(self.channels)]
        icrf = torch.stack(channelwise_icrfs, dim=0)

        icrf_model = ICRFModelDirect(n_points=self.n_points, channels=self.channels,
                                     interpolation_mode=self.interpolation_mode, initial_power=self.initial_power,
                                     icrf=icrf)

        for c in range(self.channels):
            assert torch.allclose(channelwise_icrfs[c], icrf_model.channel_params(c)[0])

    def test_icrf_model_direct_update_icrf(self):

        icrf_model = ICRFModelDirect(n_points=self.n_points, channels=self.channels,
                                     interpolation_mode=self.interpolation_mode, initial_power=self.initial_power)

        assert torch.allclose(icrf_model.icrf, self.expected_initial_icrf)

        # Manually add scalar 1.0 to all the tensors held in the nn.Parameter lists to force change.
        for p in icrf_model.direct_params:
            p.data.add_(1.0)

        # Update icrf, it should now match the changed values of the direct_params.
        icrf_model.update_icrf()
        assert torch.allclose(icrf_model.icrf, self.expected_initial_icrf + 1.0)

    def test_icrf_model_direct_forward_lookup(self):

        icrf = torch.tensor([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])

        icrf_model = ICRFModelDirect(icrf=icrf, initial_power=1.0, interpolation_mode=InterpMode.LOOKUP)

        x = torch.tensor([[[[0.0, 0.5, 1.0]], [[0.0, 0.5, 1.0]]]])  # shape (1,2,1,3)

        out = icrf_model(x)

        expected = torch.tensor([[[[0.0, 2.0, 3.0]], [[10.0, 12.0, 13.0]]]])

        assert torch.allclose(out, expected)

    def test_icrf_model_direct_forward_linear(self):

        icrf = torch.tensor([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])

        icrf_model = ICRFModelDirect(icrf=icrf, initial_power=1.0, interpolation_mode=InterpMode.LINEAR)

        x = torch.tensor([[[[0.0, 0.5, 0.75]], [[0.0, 0.5, 0.75]]]])  # shape (1,2,1,3)

        out = icrf_model(x)

        expected = torch.tensor([[[[0.0, 1.5, 2.25]], [[10.0, 11.5, 12.25]]]])

        assert torch.allclose(out, expected)

    def test_icrf_model_direct_forward_catmull(self):

        icrf = torch.tensor([[0., 1., 2., 3., 4., 5.], [10., 11., 12., 13., 14., 15.]])

        icrf_model = ICRFModelDirect(icrf=icrf, initial_power=1.0, interpolation_mode=InterpMode.CATMULL)

        x = torch.tensor([[[[0.0, 0.5, 0.75]], [[0.0, 0.5, 0.75]]]])  # shape (1,2,1,3)

        out = icrf_model(x)

        expected = torch.tensor([[[[0.0, 2.5, 3.75]], [[10.0, 12.5, 13.75]]]])

        assert torch.allclose(out, expected)

    def test_icrf_model_direct_prepare_icrf_plot_data(self):

        icrf = torch.tensor([[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]])

        icrf_model = ICRFModelDirect(icrf=icrf, initial_power=1.0, interpolation_mode=InterpMode.LINEAR)

        x, y, dydx = icrf_model._prepare_icrf_plot_data()

        # Check x is linspace [0,1]
        assert np.allclose(x, np.linspace(0, 1, 4))
        # y matches icrf
        assert np.allclose(y, icrf_model.icrf.numpy())
        # dydx = diff(y)/dx
        expected_dydx = np.diff(y, axis=1) / (x[1] - x[0])
        assert np.allclose(dydx, expected_dydx)

