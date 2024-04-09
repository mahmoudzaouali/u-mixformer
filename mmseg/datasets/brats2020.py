from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class BratsDataset(BaseSegDataset):
    """Brats2020 dataset.

    Before dataset preprocess of Brats, there are total 4 categories.
    The ``img_suffix`` is fixed to '.jpg' and
    ``seg_map_suffix`` is fixed to '.png'.

    '#440054': A dark purplish color. => IN RGB : [68, 0, 84]
    '#3b528b': A deep blue color. => IN RGB : [59, 82, 139]
    '#18b880': A teal or turquoise color. =
    '#e6d74f': A yellowish color.
    """
    METAINFO = dict(
        classes=('Background', 'Tumor'),
        palette=[[120, 120, 120], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)