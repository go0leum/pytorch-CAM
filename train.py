import torch
from torch.autograd import Variable
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
    
    writer.add_scalars('loss/train', {'train_loss':loss_avg}, (epoch))
    writer.add_scalars('accuracy/train',{'train_acc':acc_avg}, (epoch))
    
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
    
    writer.add_scalars('loss/test',{'test_loss':loss_avg}, (epoch))
    writer.add_scalars('accuracy/test',{'test_acc':acc_avg}, (epoch))
    
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