import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
noiseSteps = 1000
latent_dim = 1
learning_rate = 1e-4
batch_size = 2
numworker = 0
epochs = 1000
time_dim = 128
num_classes = 2
Lambda = 100

exp = "exp_1/"
whole_Abeta = "./data/whole_Abeta"
latent_Abeta = "./data/latent_Abeta/"
whole_MRI = "./data/whole_MRI"
path = "./data/whole_MRI/037S6046.nii"
gpus = [0]

CHECKPOINT_AAE = "result/"+exp+"AAE.pth.tar"
CHECKPOINT_Unet = "result/"+exp+"Unet.pth.tar"

train = "data_info/train.txt"
validation = "data_info/validation.txt"
test = "data_info/test.txt"