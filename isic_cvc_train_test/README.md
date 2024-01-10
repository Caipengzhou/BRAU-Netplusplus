## Dataset
To apply the model on a custom dataset, the data tree should be constructed as:
``` 
    ├── data
          ├── images
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
          ├── masks
                ├── image_1.png
                ├── image_2.png
                ├── image_n.png
```
## CSV generation 
```
python data_split_csv.py --dataset your/data/path --size 0.9 
```
## Train
```
python train.py --dataset your/data/path --csvfile your/csv/path --loss dice --batch 8 --lr 0.0005 --epoch 200 
```
## Evaluation
```
python test.py --dataset your/data/path --csvfile your/csv/path --model save_models/epoch_last.pth --debug True
```
