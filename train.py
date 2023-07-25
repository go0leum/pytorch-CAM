import torch
import numpy as np
from torch.autograd import Variable
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
        
def retrain(trainloader, model, use_cuda, epoch, criterion, optimizer,writer):

    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    i = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        # calculate accuracy
        correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        total += trainloader.batch_size
        train_acc = 100. * correct / total
        acc_sum += train_acc
        i += 1

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), train_acc))

    acc_avg = acc_sum / i
    loss_avg = loss_sum / i
    
    writer.add_scalars('loss/train', {'train_loss':loss_avg}, len(trainloader)*(epoch))
    writer.add_scalars('accuracy/train',{'train_acc':acc_avg}, len(trainloader)*(epoch))
    
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
    
    with open('result/train_acc.txt', 'a') as f:
        f.write(str(acc_avg) + '\n')

    with open('result/train_loss.txt', 'a') as f:
        f.write(str(loss_avg) + '\n')
    
    return acc_avg, loss_avg


def retest(testloader, model, use_cuda, epoch, criterion, writer):
    model.eval()
    loss_sum, acc_sum = 0, 0
    correct, total = 0, 0
    i = 0
    
    for data, target in testloader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        
        # sum up batch loss
        test_loss = criterion(output, target).item()
        loss_sum += test_loss
        
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        
        #calculate accuracy
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total += testloader.batch_size
        test_acc = 100. * correct / total
        acc_sum += test_acc
        i += 1
    
            
    acc_avg = acc_sum / i
    loss_avg = loss_sum / i
    
    writer.add_scalars('loss/test',{'test_loss':loss_avg}, len(testloader)*epoch)
    writer.add_scalars('accuracy/test',{'test_acc':acc_avg}, len(testloader)*epoch)
    
    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss_avg, correct, len(testloader.dataset), acc_avg)
    print(result)
    
    with open('result/result.txt', 'a') as f:
        f.write(result)

    with open('result/test_acc.txt', 'a') as f:
        f.write(str(test_acc) + '\n')

    with open('result/test_loss.txt', 'a') as f:
        f.write(str(test_loss) + '\n')
    
    return model.state_dict(), acc_avg, loss_avg

# total_set  = datasets.ImageFolder(data_dir)
def k_fold(fold, total_set, model, use_cuda, epoch, criterion, optimizer, batchsize,CHECK_POINT_DIR):
    
    splits = KFold(n_splits = fold, shuffle = True, random_state = 123)
    
    total_testloss, total_testacc, total_trainloss, total_trainacc = 0, 0, 0, 0
    
    for fold, (train_idx, test_idx) in enumerate(splits.split(total_set)):
        
        writer = SummaryWriter('loss_accuracy/fold_'+str(fold+1))
        
        print('Fold : {}'.format(fold+1))
        dataset_train = Subset(total_set, train_idx)
        dataset_test = Subset(total_set, test_idx)
        trainloader =DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
        testloader = DataLoader(dataset_test, batch_size=batchsize, shuffle=True)
        
        fold_testloss, fold_testacc, fold_trainloss, fold_trainacc = 0, 0, 0, 0
        
        for e in range (1, epoch + 1):
            train_acc, train_loss = retrain(trainloader ,model, use_cuda, e, criterion, optimizer, writer)
            dict, test_acc, test_loss = retest(testloader, model, use_cuda, e, criterion, writer)
            
            fold_trainloss+=train_loss
            fold_trainacc+=train_acc
            
            fold_testacc+=test_acc
            fold_testloss+=test_loss
        
        total_trainloss+=(fold_trainloss/epoch)
        total_trainacc+=(fold_trainacc/epoch)
        
        total_testloss+=(fold_testloss/epoch)
        total_testacc+=(fold_testacc/epoch)
        
        torch.save(dict, CHECK_POINT_DIR+'/' + str((fold+1) * e) + '.pt')
        
        print('Train Fold_{}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%\n'.format(fold+1, fold_trainloss/epoch, fold_trainacc/epoch))
        result='Test loss : {:.3f}, accuracy : {:.3f}\n'.format(fold_testloss/epoch, fold_testacc/epoch)
        print(result)
        
        with open('result/result.txt', 'a') as f:
            f.write(result)

        with open('result/test_acc.txt', 'a') as f:
            f.write(str(fold_testacc/epoch) + '\n')

        with open('result/test_loss.txt', 'a') as f:
            f.write(str(fold_testloss/epoch) + '\n')
        
        writer.close()
    
    print('Total Train loss : {:.3f}, accuracy : {:.3f}'.format(total_trainloss/(fold+1), total_trainacc/(fold+1)))
        
    print('\nTotal Test loss : {:.3f}, accuracy : {:.3f}'.format(total_testloss/(fold+1), total_testacc/(fold+1)))

