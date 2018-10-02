from unet import UNet

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.utils as utils
import torchvision.datasets as dataset
import torchvision.transforms as transforms

from pathlib import Path

batch_size = 5
img_size = 256
lr = 0.0002
epoch = 100

img_dir = Path("./maps/")
img_data = dataset.ImageFolder(root=img_dir, transform = transforms.Compose([
                                            transforms.Resize(size=img_size),
                                            transforms.CenterCrop(size=(img_size,img_size*2)),
                                            transforms.ToTensor(),
                                            ]))
img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True)

model = UNet(3,3,64).cuda()

try:
    model = torch.load(Path('./model/unet.pkl'))
except:
    pass

mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for i in range(epoch):
    for _,(image,label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3) 
        
        optimizer.zero_grad()

        x = torch.tensor(satel_image).cuda(0)
        y_ = torch.tensor(map_image).cuda(0)
        y = model.forward(x)
        
        loss = mse(y,y_)
        loss.backward()
        optimizer.step()

        if _ % 400 ==0:
            print(i)
            print(loss)
            utils.save_image(x.cpu().data,Path("./result/original_image_{}_{}.png".format(i,_)))
            utils.save_image(y_.cpu().data,Path("./result/label_image_{}_{}.png".format(i,_)))
            utils.save_image(y.cpu().data,Path("./result/gen_image_{}_{}.png".format(i,_)))
            torch.save(model,Path('./model/unet.pkl'))    
