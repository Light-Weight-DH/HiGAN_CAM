# python 3.7
"""Contains basic configurations for models used in this project.

Please download the public released models from the following repositories
OR train your own models, and then put them into the folder
`pretrain/tensorflow`.

PGGAN: https://github.com/tkarras/progressive_growing_of_gans
StyleGAN: https://github.com/NVlabs/stylegan
StyleGAN2: https://github.com/NVlabs/stylegan2

NOTE: Any new model should be registered in `MODEL_POOL` before used.
"""

import os

#BASE_DIR = os.path.dirname(os.path.relpath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')
PTH_MODEL_DIR = 'pytorch'
TF_MODEL_DIR = 'tensorflow'

if not os.path.exists(os.path.join(MODEL_DIR, PTH_MODEL_DIR)):
  os.makedirs(os.path.join(MODEL_DIR, PTH_MODEL_DIR))

# pylint: disable=line-too-long
MODEL_POOL = {
    # PGGAN Official.
    'pggan_celebahq': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_celebahq1024_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-celebahq-1024x1024.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'celebahq',
        'z_space_dim': 512,
        'resolution': 1024,
        'fused_scale': False,
    },
    'pggan_bedroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bedroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bedroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bedroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_livingroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_livingroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-livingroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-livingroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_diningroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_diningroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-dining_room-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-diningroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_kitchen': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_kitchen256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-kitchen-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-kitchen',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_churchoutdoor': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_churchoutdoor256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-churchoutdoor-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-churchoutdoor',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_tower': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_tower256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-tower-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-tower',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bridge': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bridge256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bridge-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bridge',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_restaurant': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_restaurant256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-restaurant-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-restaurant',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_classroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_classroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-classroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-classroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_conferenceroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_conferenceroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-conferenceroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-conferenceroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_person': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_person256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-person-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-person',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_cat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_cat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-cat-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-cat',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_dog': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_dog256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-dog-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-dog',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bird': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bird256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bird-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bird',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_horse': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_horse256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-horse-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-horse',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_sheep': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_sheep256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-sheep-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-sheep',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_cow': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_cow256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-cow-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-cow',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_car': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_car256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-car-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-car',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bicycle': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bicycle256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bicycle-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bicycle',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_motorbike': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_motorbike256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-motorbike-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-motorbike',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bus': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bus256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bus-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bus',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_train': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_train256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-train-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-train',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_boat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_boat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-boat-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-boat',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_airplane': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_airplane256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-airplane-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-airplane',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bottle': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bottle256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bottle-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bottle',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_chair': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_chair256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-chair-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-chair',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_pottedplant': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_pottedplant256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-pottedplant-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-pottedplant',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_tvmonitor': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_tvmonitor256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-tvmonitor-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-tvmonitor',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_diningtable': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_diningtable256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-diningtable-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-diningtable',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_sofa': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_sofa256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-sofa-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-sofa',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },

    # StyleGAN Official.
    'stylegan_ffhq': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_ffhq1024_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2019stylegan-ffhq-1024x1024.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'ffhq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'fused_scale': 'auto',
    },
    'stylegan_celebahq': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_celebahq1024_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2019stylegan-celebahq-1024x1024.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'celebahq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'fused_scale': 'auto',
    },
    'stylegan_bedroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_bedroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2019stylegan-bedrooms-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-bedroom',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_cat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_cat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2019stylegan-cats-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-cat',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_car': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_car512_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2019stylegan-cars-512x384.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-car',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 512,
        'fused_scale': 'auto',
    },

    # StyleGAN Self-Training.
    'stylegan_ffhq256': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_ffhq256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-ffhq-256x256-025000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'ffhq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_ffhq512': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_ffhq512_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-ffhq-512x512-025000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'ffhq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 512,
        'fused_scale': 'auto',
    },
    'stylegan_livingroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_livingroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-livingroom-256x256-030000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-livingroom',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_diningroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_diningroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-diningroom-256x256-025000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-diningroom',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_kitchen': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_kitchen256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-kitchen-256x256-030000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-kitchen',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_apartment': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_apartment256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-apartment-256x256-060000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-bedroom-livingroom-diningroom-kitchen',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_churchoutdoor': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_churchoutdoor256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-churchoutdoor-256x256-030000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-churchoutdoor',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_tower': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_tower256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-tower-256x256-030000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-tower',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_bridge': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_bridge256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-bridge-256x256-025000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-bridge',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_restaurant': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_restaurant256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-restaurant-256x256-050000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-restaurant',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_classroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_classroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-classroom-256x256-050000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-classroom',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },
    'stylegan_conferenceroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_conferenceroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan-conferenceroom-256x256-050000.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'lsun-conferenceroom',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'fused_scale': 'auto',
    },

    # StyleGAN2 Official.
    'stylegan2_ffhq': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan2_ffhq1024_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan2-ffhq-config-f.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan2_tf_official'),
        'gan_type': 'stylegan2',
        'dataset_name': 'ffhq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 1024,
        'g_architecture_type': 'skip',
        'fused_modulate': True,
    },
    'stylegan2_church': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan2_church256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan2-church-config-f.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan2_tf_official'),
        'gan_type': 'stylegan2',
        'dataset_name': 'lsun-church',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'g_architecture_type': 'skip',
        'fused_modulate': True,
    },
    'stylegan2_cat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan2_cat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan2-cat-config-f.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan2_tf_official'),
        'gan_type': 'stylegan2',
        'dataset_name': 'lsun-cat',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'g_architecture_type': 'skip',
        'fused_modulate': True,
    },
    'stylegan2_horse': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan2_horse256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan2-horse-config-f.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan2_tf_official'),
        'gan_type': 'stylegan2',
        'dataset_name': 'lsun-horse',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 256,
        'g_architecture_type': 'skip',
        'fused_modulate': True,
    },
    'stylegan2_car': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan2_car512_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan2-car-config-f.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan2_tf_official'),
        'gan_type': 'stylegan2',
        'dataset_name': 'lsun-car',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 512,
        'g_architecture_type': 'skip',
        'fused_modulate': True,
    },
}
# pylint: enable=line-too-long

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
STYLEGAN_RANDOMIZE_NOISE = False

# Settings for StyleGAN2.
STYLEGAN2_TRUNCATION_PSI = 0.5  # 1.0 means no truncation
STYLEGAN2_TRUNCATION_LAYERS = 18  # 0 means no truncation
STYLEGAN2_RANDOMIZE_NOISE = False

# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4

MAX_IMAGES_ON_RAM = 800
