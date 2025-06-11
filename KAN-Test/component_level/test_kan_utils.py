import pytest
import torch
from common_utils.kan import utils

def test_create_dataset():
    f = lambda x: torch.sum(x, dim=1, keepdim=True)
    dataset = utils.create_dataset(f, n_var=2, train_num=10, test_num=5)
    assert 'train_input' in dataset and 'test_input' in dataset
    assert dataset['train_input'].shape == (10, 2)
    assert dataset['test_input'].shape == (5, 2)
    assert dataset['train_label'].shape == (10, 1)
    assert dataset['test_label'].shape == (5, 1)
    # 检查label是否为输入之和
    assert torch.allclose(dataset['train_label'], torch.sum(dataset['train_input'], dim=1, keepdim=True), atol=1e-5)

def test_fit_params():
    x = torch.linspace(-1, 1, steps=100)
    y = 2.0 * torch.sin(3.0 * x + 1.0) + 0.5
    params, r2 = utils.fit_params(x, y, torch.sin, a_range=(2,4), b_range=(0,2), grid_number=10, iteration=1, verbose=False)
    assert params.shape == (4,)
    assert 0 <= r2 <= 1 