import torch
from PIL import Image
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt


model= models.vgg19(pretrained=True).features


# freeze the parameter that we don't use to minimize computation and memeory
for parameter in model.parameters():
    parameter.require_grad = False
    
#print(model)

# extract features
def extract_features(image, model):
    features = {}
    layers = {
#         '1'  : 'conv1_1',
#         '6'  : 'conv2_1',
#         '11' : 'conv3_1',
#         '20' : 'conv4_1',
#         '25' : 'conv4_2', # content
#         '32' : 'conv5_1'
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2', # for measuring content loss
        '28': 'conv5_1'
    }
    
    feature_image = image
    # apply singleton dimention image 
    feature_image = feature_image.unsqueeze(0)
    # pass the image to layers in the model
    for key, layer in model._modules.items():
        feature_image = layer(feature_image)
        # if the current layer in the model is part of the layers 
        # planned to use add activated feature to the features dictionary
        if key in layers:
            features[layers[key]] = feature_image
            #print(feature_image)
            
    return features


# do some transformation to reduce the size of the image and be able to pass it to the 
# netowrk

transform = transforms.Compose([transforms.Resize(300),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5, 0.5),(0.5,0.5,0.5))])
    
def tensor_to_Img(tensor):
    img = tensor.clone().detach().numpy().squeeze()
    img = img.transpose(1,2,0)
    img = img*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return img
    

def img_show(content, style):
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(tensor_to_Img(content), label = "Content")
    ax2.imshow(tensor_to_Img(style), label = "Style")
    plt.show()
    
    
def gram_matrix(feature):
    t, depth, height, weight = feature.size()
    #input = torch.Tensor(2, 4, 3) # input: 2 x 4 x 3
    #print(input.view(1, -1, -1, -1).size()) # prints - torch.size([1, 2, 4, 3])
    
    feature = feature.view(depth, height*weight)
    # multiply the matrix with its transpose
    gram_matrix = torch.mm(feature, feature.t())
    
    return gram_matrix
    

def create_content_loss(content, output_features):
    
    content_features = extract_features(content,model)
    content_loss = torch.mean((content_features['conv4_2']-output_features['conv4_2'])**2)
    return content_loss
    
    
    
    
def create_style_loss(style, output_features):
    
    style_features = extract_features(style,model)
    style_weight = {"conv1_1" : 1.0, 
                    "conv2_1" : 0.8,
                    "conv3_1" : 0.4,
                    "conv4_1" : 0.2,
                    "conv5_1" : 0.1}
    # compute the correlation (gram matrix)
    gram_layers = {}
    for layer in style_features:
        gram_layers[layer]= gram_matrix(style_features[layer]) 
    
    # calculate style loss
    style_loss = 0
    for layer in style_weight:
        style_gram = gram_layers[layer]
        
        output_gram = output_features[layer]  
        # used for normalization
        _,depth, weight, height = output_gram.shape
        output_gram = gram_matrix(output_gram)
        # squared mean of the output and the style gram
        style_loss += (style_weight[layer]*torch.mean((output_gram-style_gram)**2))/depth*weight*height
        
        
    return style_loss
        


    
def main():
    # alpha
    content_wt = 100
    # beta
    style_wt = 1e5
    style_wt2 = 1e6
    style_wt3 = 1e6
    
    print_after = 100
    epochs = 400
    
    content = Image.open("cosco.jpg").convert("RGB")
    content = transform(content)
    
    style = Image.open("1-style.jpg").convert("RGB")
    style = transform(style)
    
    style2 = Image.open("2-style1.jpg").convert("RGB")
    style2 = transform(style2)
    
    style3 = Image.open("style.jpg").convert("RGB")
    style3 = transform(style3)
    
    
    img_show(content, style)
    img_show(style3, style2)
    
    #output is initalized from the content first
    output = content.clone().requires_grad_(True)
    
    # optimize the output image
    # this is a hyper parameter 
    optimizer = torch.optim.Adam([output],lr=0.001)
    # lr - learning rate, 0.07
    # activate output features
    for i in range(1, epochs + 1):
        output_features = extract_features(output, model)
    
        content_loss = create_content_loss(content, output_features)
        style_loss = create_style_loss(style, output_features)
        style_loss2 = create_style_loss(style2, output_features)
        style_loss3 = create_style_loss(style3, output_features)
        
        total_loss = content_wt*content_loss + style_wt*style_loss + style_wt2*style_loss2 + style_wt3*style_loss3 
    
        if i% 10 == 0:
            print("epoch ", i, " ", total_loss)
      
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
        if i % print_after == 0:
            plt.imshow(tensor_to_Img(output),label="Epoch "+str(i))
            plt.show()
           # plt.imsave(str(i)+'.png',tensor_to_Img(output),format='png')

if __name__ == '__main__':
    main()