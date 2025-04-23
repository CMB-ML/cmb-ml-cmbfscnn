import cmbfscnn.CNN_models as cmbfscnn_models
from cmbml.utils.suppress_print import SuppressPrint

# def make_cmbfscnn(level=3, in_channels=None, out_channels=1, n_feats=16):
def make_cmbfscnn(model_dict, cmbfscnn_level):
    if cmbfscnn_level == 3:
        model_class = cmbfscnn_models.CMBFSCNN_level3
    elif cmbfscnn_level == 4:
        model_class = cmbfscnn_models.CMBFSCNN_level4

    net = model_class(**model_dict)

    # with SuppressPrint():
        # net = model_class(**model_dict)
    return net