def get_vlm_model(config):
    from .QWen2_5 import _QWen_VL_Interface
    return _QWen_VL_Interface(config)
