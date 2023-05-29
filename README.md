# Loss-Landscape-Reshaping


## Training Commands
### ImageNet
ResNet50:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./distributed_train1.sh 8  /home/bdc/datasets/ImageNetLMDB/ -b 256 --model resnet50  --givenA 1 0 --givenB -1 0 --ConverOrd 1 --notes CosConOrd1PreAct1Default --sched cosine --epochs 90 --lr 0.8 --dist-bn reduce --warmup-epochs 10 --cooldown-epochs 0 --pin-mem -j 4 --settings Default --IniDecay 0.7 --norm_loss --dls_act sech --dls_coe0 2.5 --dls_coe1 0.2 --seed 42 --setting DLS --rep_num 0 --use-multi-epochs-loader --end 1 > /media/bdc/clm/DeformingTheLossSurface/ImageNet/log/resnet50_dls_coe0_2p5_coe1_0p2sech_FP32.txt 2>&1
```
### CIFAR
Run with comet-ml:
```python
CUDA_VISIBLE_DEVICES=0 comet optimize train_tuning.py single.json > tuning0.txt 2>&1 &
```
