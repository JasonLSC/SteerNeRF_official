import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_SR', type=float, default=1e-4,
                        help='learning rate of super-resolution')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # dataset making options
    parser.add_argument('--render_low_res', action='store_true', default=False,
                        help='use to generate low_res high_res pairs')
    parser.add_argument('--render_downsample', type=float, default=1,
                        help='use to generate low_res high_res pairs')

    # super sampling options
    parser.add_argument('--super_sampling', action='store_true', default=False,
                        help='whether to do supersampling')
    parser.add_argument('--super_sampling_factor', type=int, default=4,
                        help='upsample factor(>=1) for rendered frame')
    parser.add_argument('--sr_model_type', type=str, default='stylegan_sr_res',
                        help='model type of super-resolution')

    parser.add_argument('--patch_size', type=int, default=150,
                        help='height and width of the training patch')
    parser.add_argument('--frame_num', type=int, default=1,
                        help='number of input frames for upsampling')

    parser.add_argument('--feature_training', action='store_true', default=False,
                        help='whether to take NeRF feature as input of SR model ')

    # multiple stage options
    parser.add_argument('--training_stage', type=str, default='NeRF_pretrain',
                        help='to specify training stage: NeRF_pretrain, End2End')
    parser.add_argument('--complete_pipeline', action='store_true', default=False,
                        help='automatically doing sr_pretrain and joint training')
    parser.add_argument('--direct_E2E', action='store_true', default=False,
                        help='No need of SR_pretrain, just End2end training directly')
    parser.add_argument('--dataset', type=str, default='Synthetic_NeRF',
                        help='to specify dataset name: Synthetic_NeRF, LLFF, TanksAndTemple')

    # NGP num of param. options
    parser.add_argument('--log2_T', type=int, default=19,
                        help='log2 of length of hash table')

    # inference optiopns
    parser.add_argument('--render_traj', action='store_true', default=False)
    parser.add_argument('--save_traj_img', action='store_true', default=False)
    parser.add_argument('--TRT_enable', action='store_true', default=False)
    parser.add_argument('--TRT_engine_file', type=str,
                        help='file of TRT engine')


    return parser.parse_args()
