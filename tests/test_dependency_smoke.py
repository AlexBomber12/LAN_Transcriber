def test_omegaconf_importable():
    from omegaconf import OmegaConf

    assert hasattr(OmegaConf, "create")
