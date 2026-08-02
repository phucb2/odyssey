def test_ml_public_imports() -> None:
    from odyssey.models import create_cnn_model
    from odyssey.training import Learner

    model = create_cnn_model()
    assert model is not None
    assert Learner is not None


def test_torch_compile_available_on_windows() -> None:
    import sys

    from odyssey.training.cuda import torch_compile_available

    if sys.platform == "win32":
        assert torch_compile_available() is False
