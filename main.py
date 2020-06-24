import torch 
import numpy as np 
import os
from data_loader import *
from arguments import *
import torchvision
import torchvision.transforms as transforms
from model import *


def train(args, model, loss_fn, optim, train_loader):
    Loss_log=[]
    model.train()
    for i, (img, label) in enumerate(train_loader):
        if args.cuda:
            img= img.cuda()
            label= label.cuda()
        
        output= model(img)
        # compute loss:
        loss= loss_fn(output, label)
        optim.zero_grad()
        # backpropagation;
        loss.backward()
        # gradient descent steo:
        optim.step()
        Loss_log.append(loss.item())
    Loss_log= np.asarray(Loss_log)
    return np.mean(Loss_log), np.std(Loss_log)


def val(args, model, loss, val_loader):
    mse= nn.MSELoss()
    if args.cuda:
        mse=mse.cuda()
    model.eval()
    Loss_log=[]
    for i, (img, label) in enumerate(val_loader):
        if args.cuda:
            img= img.cuda()
            label= label.cuda()
        
        with torch.no_grad():
            output= model(img)
        error= mse(output, label)
        Loss_log.append(error.item())
    Loss_log= np.asarray(Loss_log)
    return np.mean(Loss_log)

    

def test(args, model, bce_loss, test_loader):
    l1= nn.L1Loss()

    correct=0
    total=0
    model.eval()
    Loss_log_l1, Loss_log_BCE=[], []
    for i, (img, label) in enumerate(test_loader):
        if args.cuda:
            img= img.cuda()
            label= label.cuda()
        with torch.no_grad():
            output= model(img)
        total+= label.size(0)
        correct+= (output == label).sum().item()
        print(output, label)
        l1_value= l1(output, label)
        bce_value= bce_loss(output, label)
        # Evaluation metrics for submission
        Loss_log_l1.append(l1_value.item())
        Loss_log_BCE.append(bce_value.item())
    Loss_log_l1= np.asarray(Loss_log_l1)
    Loss_log_BCE= np.asarray(Loss_log_BCE)
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    print('l1 error:{:.4f}, BCE error:{:.4f}'.format(np.mean(Loss_log_l1), np.mean(Loss_log_BCE)))



def main():
    args= get_args()

    # load the data
    _transforms= transforms.Compose([transforms.Resize((256,256)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ToTensor()])
    if args.train:
        covid_train_data= CovidLoader(args, split='train', transforms= _transforms)
        covid_val_data= CovidLoader(args, split='val', transforms=_transforms)
        # convert to an iterator;
        covid_train_loader= torch.utils.data.DataLoader(covid_train_data, shuffle=True, batch_size= args.batch_size, num_workers=args.workers)
        covid_val_loader= torch.utils.data.DataLoader(covid_val_data, shuffle=True, batch_size= args.batch_size, num_workers=args.workers)
        print('training data size: {}'.format(len(covid_train_data)))
    
    if args.test:
        covid_test_data= CovidLoader(args, split='test', transforms=_transforms)
        covid_test_loader= torch.utils.data.DataLoader(covid_test_data, shuffle=False, batch_size= args.batch_size, num_workers=args.workers)
        print('testing data size: {}'.format(len(covid_test_data)))
    
    # create the model
    model = SimpleModel()
    # extra model prepping
    loss= nn.BCELoss()
    
    if args.cuda:
        model= model.cuda()
        loss= loss.cuda()

    if args.test:
        state_dict= torch.load('/home/mukund/Documents/COVID/models/epoch_10.pth')
        model.load_state_dict(state_dict)
        print('model weights loaded')


    optimizer= torch.optim.Adam(model.parameters())
    # training:
    if args.train:
        print('training started')
        for e in range(args.epochs):
            loss_value, std_dev= train(args, model, loss, optimizer, covid_train_loader)
            #---- print stats;
            print('epoch:{}, train loss:{:.3f}'.format(e, loss_value))
        
            if e % args.eval_interval==0:
                eval_loss= val(args, model, loss, covid_val_loader)
                print('epoch:{}, eval loss: {:.3f}'.format(e, eval_loss))
            
            if e % args.save_interval==0:
                file_name=r'./models/epoch_{}.pth'.format(e)
                torch.save(model.state_dict(), file_name)
    # test:
    if args.test:
        print('testing started')
        test(args, model, loss, covid_test_loader)

if __name__=='__main__':
    main()
