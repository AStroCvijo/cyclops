import torch

from cyclops.utils.device import dataloader_workers, pin_memory_for, resolve_device


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_pin_memory_only_cuda():
    assert pin_memory_for(torch.device("cuda")) is True
    assert pin_memory_for(torch.device("mps")) is False
    assert pin_memory_for(torch.device("cpu")) is False


def test_resolve_device_auto_picks_accelerator(monkeypatch):
    monkeypatch.setattr("cyclops.utils.device._mps_available", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"
