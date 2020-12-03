import torch 
#torchvision package consists of popular datasets, model architectures, and common image transformations for computer vision.
from torchvision import transforms , models 
# Python Imaging Library(PIL) is a free library
# for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
# for content comparison we use activation/feature map at some deeper layer(L)- 1 or 2 layers prior to the output(softmax) layer. 
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np

device = ("cuda" if torch.cuda.is_available() else "cpu")

# why vgg19- because this newtork has deeper layers
# Since the image classification convolutional neural networks like VGG16 are 
# forced to learn high-level features /abstract representations or the “content” of images at the deeper layer, 
# vgg works better because it has a lot of layers and it learns in  different scales and it caputres more details about the image
# others down sample off sample alot so they are not a good fit
model = models.vgg19(pretrained=True).features
# requires_grad = false -do not update (freeze) parts of the network
#  simply avoids unnecessary computation, update, and storage of
#gradients at those nodes and does not create subgraphs which saves memory.
for p in model.parameters():
    p.requires_grad = False
model.to(device)
print(model._modules.items())
# the first four layers are for measureing content loss and the last one is for style loss

# given an input and a model it iterates through each of the layers of the model and keep passing the input and calculate
# the out put and if it is found in the dictionary    
def model_activations(input,model):
    layers = {
    '0' : 'conv1_1',
    '5' : 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '21': 'conv4_2', # for measuring content loss
    '28': 'conv5_1'
    }
    features = {}
    x = input
    # input = torch.Tensor(2, 4, 3) # input: 2 x 4 x 3
    # print(input.unsqueeze(0).size()) # prints - torch.size([1, 2, 4, 3])
    # insert the singleton dimension  without explicitly being aware some of the dimentions
    x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    
    return features



# resize the image- since we aren't using gpu we can't handle

transform = transforms.Compose([transforms.Resize(300),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


content = Image.open("suny.jpg").convert("RGB")
content = transform(content).to(device)
print("COntent shape => ", content.shape)
style = Image.open("ex.jpg").convert("RGB")
style = transform(style).to(device)
# image convert it takes a tensor and converts into printable form in mat.lib
def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(imcnvt(content),label = "Content")
ax2.imshow(imcnvt(style),label = "Style")
plt.show()
# correlation 
def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)
    gram_mat = torch.mm(imgfeature,imgfeature.t())
    
    return gram_mat

# the desired output which its size is the same as its content
#requires_grad= gradient required because optimization is required for output img
target = content.clone().requires_grad_(True).to(device)

#set device to cuda if available
print("device = ",device)

# capture the style and content from the desired layers and safes those features
# it is done outside the optimization as nothing will be done to the features
# the output image is the one that is optimized
style_features = model_activations(style,model)
content_features = model_activations(content,model)

#0.4, 0.3,  0.2, 0.2,0.2 experminent with this
# how much wait each layer will have with the style and content
# the style layers for the lower layers should be higher
# style loss - 
# 
style_wt_meas = {"conv1_1" : 1.0, 
                 "conv2_1" : 0.8,
                 "conv3_1" : 0.4,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1}
# pass each layer to gram matrix function and calculate its 
style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features}
# generally style wait should wait more than the content
content_wt = 100
style_wt = 1e8

print_after = 500
epochs = 4000
# this is a hyper parameter 
optimizer = torch.optim.Adam([target],lr=0.007)

for i in range(1,epochs+1):
    target_features = model_activations(target,model)
    content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_wt_meas:
        style_gram = style_grams[layer]
        target_gram = target_features[layer]
        _,d,w,h = target_gram.shape
        target_gram = gram_matrix(target_gram)

        style_loss += (style_wt_meas[layer]*torch.mean((target_gram-style_gram)**2))/d*w*h
    # conten wt and style wt are alpha and gamma the rest is just a hyper parameter
    total_loss = content_wt*content_loss + style_wt*style_loss 
    
    if i%10==0:       
        print("epoch ",i," ", total_loss)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if i%print_after == 0:
        plt.imshow(imcnvt(target),label="Epoch "+str(i))
        plt.show()
        plt.imsave(str(i)+'.png',imcnvt(target),format='png')

# don't go more than a 500 iterations cuz it will get 




