from PIL import Image
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn.functional as F

pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.PILToTensor()
writer = SummaryWriter("./logs")
images_path = "./images"
images_names = os.listdir(images_path)
torch_resize = Resize([200, 200])
images = [Image.open(i).convert("1") for i in [os.path.join(images_path, j) for j in os.listdir(images_path)]]

for i in images:
    pass
    #i.show()
    #检验是否转化为二值化图像

tensor_images = [pil_to_tensor(i) for i in images] #转化为tensor类型
tensor_images = [torch_resize(i) for i in tensor_images] #resize成200x200

#卷积核
kernel = torch.tensor([[1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]], dtype=torch.float)
kernel = torch.reshape(kernel, (1, 1, 4, 4))

def meanValue(input): #卷积
    input = input.clone().detach()
    return F.conv2d(input, kernel, stride=1)

def pool(input):
    image = input.clone().detach()
    for i in range(2, len(image.data[0][0])-2):
        for j in range(2, len(image.data[0][0][i])-2):
            image.data[0][0][i][j] = min([input.data[0][0][x][y] for x in range(i-2, i+3) for y in range(j-2, j+3)])
    return image

def restore1(input):
    image = input.clone().detach()
    maxnum = 0
    for i in range(len(image.data[0][0])):
        for j in range(len(image.data[0][0][i])):
            if image.data[0][0][i][j] > maxnum:
                maxnum = image.data[0][0][i][j]

    for i in range(len(image.data[0][0])):
        for j in range(len(image.data[0][0][i])):
            image.data[0][0][i][j] = image.data[0][0][i][j]/maxnum

    return image

def findMid(image):#寻找中值
    return int((min(image)+max(image))/2)

def distinct(image):
    ans = image.clone().detach()
    for i in range(len(image.data[0][0])):
        for j in range(len(image.data[0][0][i])):
            if image.data[0][0][i][j] < 0.97:
                ans.data[0][0][i][j] = 0
            else:
                image.data[0][0][i][j] = 1.0
    return ans

def get_xlist(image):
    lx = [0 for i in range(len(image.data[0][0][0]))]
    s = 0
    for j in range(len(image.data[0][0][0])):
        s = 0
        for i in range(len(image.data[0][0])):
            if image.data[0][0][i][j] > 0.5:
                s += 1
        lx[j] = s
    return lx

def get_ylist(image, x1, x2):
    ly = [0 for i in range(x2 - x1 + 1)]
    s = 0
    for i in range(x2 - x1 + 1):
        s = 0
        for j in range(len(image.data[0][0][i])):
            if image.data[0][0][i][j] > 0.5:
                s += 1
        ly[i] = s
    return ly

def get_fenge(lx):
    lister_sort = lx.copy()
    lister_sort.sort()
    recommend_num = lister_sort[int(len(lister_sort)*0.55)]
    lx = [i-recommend_num for i in lx]
    for i in range(len(lx)):
        if lx[i] < 0:
            lx[i] = 0
    op = 0
    ed = 0
    sum = 0
    ans = []
    while ed < len(lx) and op < len(lx):
        if lx[op] == 0:
            op+=1
            ed = op
        else:
            if lx[ed] > 0:
                ed += 1
            else:
                ans.append(lx[op:ed].copy())
                op = ed
        if ed == len(lx) and lx[ed - 1] > 0:
            ans.append(lx[op:].copy())
    print(ans)
    print(f"有{len(ans)}个瓶子")
    for i in ans:
        if max(i) > 31:
            sum+=1
    print(f"有{sum}个瓶子不合格")


img = torch.reshape(tensor_images[0],[1,200,200])
writer.add_image(f"image0", img)
for i in range(len(tensor_images)):
    alt = tensor_images[i]
    alt = torch.reshape(alt, (1, 1, 200, 200))
    alt = distinct(alt)
    alt = meanValue(alt)
    alt = pool(alt)
    alt = restore1(alt)
    alt = distinct(alt)
    writer.add_images(f"image{i+1}", alt, global_step=i)
    lx = get_xlist(alt)
    get_fenge(lx)
writer.close()








