import torch
from typing import Literal
from kornia.utils.image import perform_keep_shape_image
from kornia.color import rgb_to_lab, lab_to_rgb


# @perform_keep_shape_image
def white_balance(
    input: torch.Tensor,
    mode: Literal['grayworld'],
    do_not_care_for_grad: bool = False
):
    if mode == 'grayworld':
        assert input.shape[-3] == 3
        input_lab = rgb_to_lab(input)
        avg = input_lab[:, 1:, ...]
        if not do_not_care_for_grad:
            avg = avg.clone() # clone is required to compute grad
        avg = avg.mean(dim = (-1, -2), keepdim=True) 
        input_lab[:, 1:, ...] = input_lab[:, 1:, ...] - avg * (input_lab[:, :1, ...] / 100)  * 1.1
        return lab_to_rgb(input_lab)
    else:
        raise ValueError(f"Unknown white_balance mode: {mode}")

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    t = torch.rand((18, 3, 100, 100), dtype=torch.float32)
    t.requires_grad = True
    b = white_balance(t, mode = 'grayworld')
    c = b.sum()
    c.backward()
    # print(b.grad_fn, b.grad) 