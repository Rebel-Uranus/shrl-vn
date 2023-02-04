# VTNet: Visual Transformer Network for Object Goal Navigation


## Install

## Low_pretrain
```python
python main.py --gpu-ids 0 1 2 3 --workers 16 --model VTNetModel --detr --title a3c_22cls_vistrans_base --work-dir ./work_dirs/ --test-after-train --pretrained-trans ./... --continue-training ./... --train-mode search
```
## high_pretrain
```python
python main.py --gpu-ids 0 1 2 3 --workers 16 --model VTNetModel --detr --title a3c_22cls_vistrans_base --work-dir ./work_dirs/ --test-after-train --continue-training ./... --train-mode high_pretrain
```
## fine-tune
```python
python main.py --gpu-ids 0 1 2 3 --workers 16 --model VTNetModel --detr --title a3c_22cls_vistrans_base --work-dir ./work_dirs/ --test-after-train --continue-training ./... --train-mode fine-tune
```

## Pretraining

```python
python main_pretraining.py --gpu-ids 0 1 --workers 4 --model BaseModel --detr --title a3c --work-dir ./pretrain_dirs/ 
```

## Testing

```python
python full_eval.py --gpu-ids 2 3 --detr --save-model-dir ./work_dirs/a3c_previstrans_base_train_2020-08-17_03:19:07/trained_models/ --results-json ./work_dirs/a3c_previstrans_base_train_2020-08-17_03:19:07/result.json --model VisualTransformerModel --title a3c_previstrans_base
```

## Testing

```python
tensorboard --logdir=./work_dirs/runs/a3c_22cls_vistrans_base_train_2021-06-11_10-41-58/ --port 6006
```

python main.py --gpu-ids 0 1 --workers 8 --model VTNetModel --detr --title test_visual --work-dir ./work_dirs/ --test-after-train --continue-training ./... --max-ep 10 --visualize-file-name visualize_action_list.json

