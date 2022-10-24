## Requirements

The code requires *open3d* 

## 3D Object Classification

### train on ModelNet40
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir data/ModelNet40/ --run_mode train --mvnetwork viewgcn --nb_views 12 --views_config spherical --dset_variant hardest --pc_rendering --object_color custom --epochs 35 --batch_size 20 --viewgcn_phase all --exp_id 0000 --depthmap fix --points_radius 0.007 --canonical_distance 1.0 --first_stage_epochs 30 --depth 18 --use_avgpool
```

### train on ScanObjectNN
```
CUDA_VISIBLE_DEVICES=0 python main_pointmlp.py --data_dir data/ScanObjectNN/ --run_mode train --mvnetwork viewgcn --nb_views 20 --views_config spherical --dset_variant hardest --pc_rendering --object_color custom --epochs 100 --batch_size 20  --viewgcn_phase all --exp_id 0001 --depthmap fix --points_radius 0.007 --canonical_distance 1.0 --first_stage_epochs 30 --depth 18
```

[comment]: <> (## 3D Shape Retrieva)

[comment]: <> (### train on ShapeNetCore55)

[comment]: <> (```)

[comment]: <> (CUDA_VISIBLE_DEVICES=0 python main_pointmlp.py --data_dir data/ScanObjectNN/ --run_mode train --mvnetwork viewgcn --nb_views 20 --views_config spherical --dset_variant hardest --pc_rendering --object_color custom --epochs 100 --batch_size 20  --viewgcn_phase all --exp_id 0001 --depthmap fix --points_radius 0.007 --canonical_distance 1.0 --first_stage_epochs 30 --depth 18)

[comment]: <> (```)

[comment]: <> (### train on ModelNet40)

[comment]: <> (```)

[comment]: <> (CUDA_VISIBLE_DEVICES=0 python main.py --data_dir data/ModelNet40/ --run_mode train --mvnetwork viewgcn --nb_views 12 --views_config spherical --dset_variant hardest --pc_rendering --object_color custom --epochs 35 --batch_size 20 --viewgcn_phase all --exp_id 0000 --depthmap fix --points_radius 0.007 --canonical_distance 1.0 --first_stage_epochs 30 --depth 18 --use_avgpool)

[comment]: <> (```)


## Dataset

ModelNet40, ScanObjectNN, ShapeNet55
