import torch
from matplotlib import colormaps    # type: ignore


class Colormap(torch.nn.Module):
    """
    A PyTorch module for applying a colormap to a tensor.

    Attributes
    ----------
    colors : torch.Tensor
        The tensor of colors for the colormap, 256 float3 values.

    Parameters
    ----------
    cmap : str, optional
        The name of the colormap to use. Defaults to "turbo".
    """

    colors: torch.Tensor

    def __init__(self, cmap: str = "turbo"):
        super().__init__()
        colors = torch.tensor(colormaps[cmap].colors)
        self.register_buffer("colors", colors)

    @torch.no_grad()
    def forward(self, tensor: torch.Tensor, clip: bool = True) -> torch.Tensor:
        """
        Applies the colormap to the input tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to which the colormap is applied.
        clip : bool, optional
            If True, clips the input tensor to the range [0, 1] before applying the colormap.
            Defaults to True.

        Returns
        -------
        torch.Tensor
            The tensor with the colormap applied.
        """
        tensor = torch.nan_to_num(tensor, 0)
        if clip:
            tensor = torch.clip(tensor, min=0.0, max=1.0)
        else:
            if (tensor < 0.0 or tensor > 1.0).any():
                min, max = tensor.min(), tensor.max()
                raise RuntimeError("Colormap expects normalized values in range [0,1]. "
                                   f"Given tensor exceeds this limit: min={min}, max={max}")
        index = (tensor * 255.).long()
        return self.colors[index]
