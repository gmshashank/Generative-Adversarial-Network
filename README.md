# Generative-Adversarial-Network
To train a Generative Adversarial Network (GAN) on a custom dataset of Red CAR image. Deploy the model using Streamlit and AWS Lambda

## Dataset
‪[Dataset](assets/car_dataset.png)

The dataset was created with 1463 images of red cars collected from internet.

The original Original CAR dataset is hosted on Amazon S3. It can be downloaded from following URL
https://datasets-sgm.s3.eu-west-3.amazonaws.com/car_images.zip

The resized CAR dataset (size = 100 x 100) is hosted on Amazon S3. It can be downloaded from following URL
https://datasets-sgm.s3.eu-west-3.amazonaws.com/car_images_100x100.zip

The dataset has mean and standard deviation as follows:
```bash
mean=[0.570838093757629, 0.479552984237671, 0.491760671138763],
std=[0.279659748077393, 0.309973508119583, 0.311098515987396],
```

# Model Summary


```
DCGAN model 


  | Name          | Type          | Params
------------------------------------------------
0 | generator     | Generator     | 3.6 M 
1 | discriminator | Discriminator | 2.0 M 
2 | criterion     | BCELoss       | 0     
------------------------------------------------
5.6 M     Trainable params
0         Non-trainable params
5.6 M     Total params
22.462    Total estimated model params size (MB)
```

#
## Training

The training can be excuted using the following command
```bash
python run.py --data_dir=data/ --batch_size=128 --gpus=1 --num_workers=8 --image_size=128 --num_epochs=2000
```

Here, used `spectral_norm` instead of the normal batch_norm2d, Please refer this paper [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)

#
## Loss
```nn.BCELoss()``` is used for training the model

![GAN Loss](assets/gan_loss.png)

#
## Visualization of Generator's progression
![Visualization of Generator's progression](assets/gan_animation.gif)

#
## Real Images vs. Fake Images
![Real Images vs. Fake Images](assets/gan_generated_comparison.png)
