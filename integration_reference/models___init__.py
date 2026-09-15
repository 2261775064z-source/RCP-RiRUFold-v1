from .deepunfolding import *
from .RiRFold import RiRFold
from .rirufold_admm import RiRFoldADMM
from .rirufold_rcp import build_rcp_model
from .rirufold_cmc import build_cmc_model
from .unfolding_ablation import ADMMLikeUnfolding, RiRFoldNoFeedback

def get_model(name, net=None, **kwargs):
    if name == 'rpcanet':
        net = RPCANet9(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32)

    elif name == 'rpcanet_pp':
        net = RPCANet_LSTM(stage_num=6, slayers=6, mlayers=3, channel=32)

    elif name == 'admm_unfolding':
        net = ADMMLikeUnfolding(stage_num=6)

    elif name == 'rirufold':
        net = RiRFold(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32)

    elif name == 'rirufold_admm':
        net = RiRFoldADMM(
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
            use_svt=kwargs.get('use_svt', True),
        )

    elif name == 'rirufold_admm_nofeedback':
        net = RiRFoldADMM(
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
            use_feedback=False,
        )

    elif name == 'rirufold_admm_rawwls':
        net = RiRFoldADMM(
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
            structure_source='input',
        )

    elif name == 'rirufold_admm_pointwse':
        net = RiRFoldADMM(
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
            structure_source='none',
        )

    elif name.startswith('rirufold_rcp'):
        net = build_rcp_model(
            name,
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
        )

    elif name.startswith('rirufold_cmc'):
        net = build_cmc_model(
            name,
            stage_num=kwargs.get('stage_num', 5),
            hidden_channels=kwargs.get('hidden_channels', 24),
        )

    elif name == 'rirufold_nofeedback':
        net = RiRFoldNoFeedback(stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32)

    elif name == 'rpcanet_pp_s3':
        net = RPCANet_LSTM(stage_num=3, slayers=6, mlayers=3, channel=32)

    elif name == 'rpcanet_pp_s9':
        net = RPCANet_LSTM(stage_num=9, slayers=6, mlayers=3, channel=32)
    else:
        raise NotImplementedError

    return net
