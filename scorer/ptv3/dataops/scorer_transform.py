from scorer.ptv3.dataops.transform import Compose, TRANSFORMS


class TEST_TRANSFORM():
    def __init__(self) -> None:
        pre_transform = [
            dict(type="NormalizeCoord"),
        ]
        voxelize = dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                keys=("coord", "pc_idx"),
                return_grid_coord=True,
            )
        post_transform = [
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "pc_idx"),
                feat_keys=["coord", "pc_idx"],
            )
        ]
        self.pre_transform = Compose(pre_transform)
        self.voxelize = TRANSFORMS.build(voxelize)
        self.post_transform = Compose(post_transform)

    def __call__(self, coord, pc_idx):
        data_dict = dict(
            coord=coord,
            pc_idx=pc_idx,
        )
        data_dict = self.pre_transform(data_dict)
        data_dict = self.voxelize(data_dict)
        data_dict = self.post_transform(data_dict)
        return data_dict