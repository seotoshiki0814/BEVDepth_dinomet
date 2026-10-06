import mmcv
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from gen_info import generate_info


def main():
    dataroot = './data/nuScenes/'

    nusc = NuScenes(
        version='v1.0-mini',
        dataroot=dataroot,
        verbose=True
    )

    # nuScenes mini:
    # 8 scenes for mini_train
    # 2 scenes for mini_val
    train_infos = generate_info(
        nusc,
        splits.mini_train
    )

    val_infos = generate_info(
        nusc,
        splits.mini_val
    )

    mmcv.dump(
        train_infos,
        './data/nuScenes/nuscenes_infos_train.pkl'
    )

    mmcv.dump(
        val_infos,
        './data/nuScenes/nuscenes_infos_val.pkl'
    )

    print('train infos:', len(train_infos))
    print('val infos  :', len(val_infos))


if __name__ == '__main__':
    main()