import argparse
import logging
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

#custom module
from models import get_models
from dataset import STDataset
from utils import adjust_lr, cosine_decay_with_warmup, set_grad

dir_img1 = r'D:\tsungyu\chromosome_data\cyclegan_data\real_img\real_zong' #訓練集的圖片所在路徑 長庚圖片
dir_style1 = r'D:\tsungyu\chromosome_data\cyclegan_data\fake_img\fake_chang' #訓練集的真實label所在路徑 長庚圖片榮總風格
dir_img2 = r'D:\tsungyu\chromosome_data\cyclegan_data\real_img\fake_zong' #訓練集的圖片所在路徑 長庚圖片
dir_style2 = r'D:\tsungyu\chromosome_data\cyclegan_data\fake_img\real_chang' #訓練集的真實label所在路徑 長庚圖片榮總風格
dir_checkpoint = r'log\CE_m0999' #儲存模型的權重檔所在路徑
teacher_load_path = r'D:\tsungyu\AdaIN_domain_adaptation\log\onlyASL\unet_50.pth'

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--total_epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--warmup_epoch',type=int,default=0,help='warm up the student model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--init_lr','-r',type = float, default=2e-2,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    parser.add_argument("--momentum", "-m", type=float, default=0.999, help="momentum parameter for updating teacher model.")
    parser.add_argument('--loss', type=str,default='cross_entropy',help='loss metric, options: [kl_divergence, cross_entropy, dice_loss]')
    parser.add_argument('--pad_mode', action="store_true",default=False, help='unet used crop or pad at skip connection') # pretrained model , pad mode == True
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset1 = STDataset(source_dir = dir_img1, style_dir= dir_style1)
    trainingDataset2 = STDataset(source_dir = dir_img2, style_dir= dir_style2)
    trainingDataset = ConcatDataset([trainingDataset1, trainingDataset2])
    os.makedirs(dir_checkpoint,exist_ok=False)

    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(dir_checkpoint,"log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    student = get_models(model_name="student_unet", is_cls=True,args=args)
    teacher = get_models(model_name="teacher_unet", is_cls=True,args=args)

    pretrained_model_param_dict = torch.load(teacher_load_path)
    student_param_dict = student.state_dict()
    teacher_param_dict = teacher.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict_s = {k: v for k, v in pretrained_model_param_dict.items() if k in student_param_dict}
    pretrained_dict_t = {k: v for k, v in pretrained_model_param_dict.items() if k in teacher_param_dict}
    # 2. overwrite entries in the existing state dict
    student_param_dict.update(pretrained_dict_s)
    teacher_param_dict.update(pretrained_dict_t)
    # 3. load the new state dict
    # student.load_state_dict(student_param_dict)
    teacher.load_state_dict(teacher_param_dict)
    
    logging.info(student)
    logging.info("="*20)
    logging.info(teacher)
    optimizer = torch.optim.Adam(student.parameters(),lr = args.init_lr,betas=(0.9,0.999))
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    student and teacher model are both initialized by zong weights.
    update teacher model(EMA) at every \'epoch\' too. 
    
    dir_img1: {dir_img1}
    dir_style1: {dir_style1}
    dir_img2: {dir_img2}
    dir_style2: {dir_style2}
    dir_checkpoint: {dir_checkpoint}
    teacher_load_path : {teacher_load_path}
    args: 
    {args}
    =======================================
    ''')
    try:
        training(net=student,
                 teacher=teacher,
                optimizer = optimizer,
                dataset = trainingDataset,
                args=args,
                save_checkpoint= True,)
                
    except KeyboardInterrupt:
        torch.save(student.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    return

def training(net,
             teacher,
             optimizer,
             dataset,
             args,
             save_checkpoint: bool = True):

    arg_loader = dict(batch_size = args.batch_size, num_workers = 4)
    train_loader = DataLoader(dataset,shuffle = True, **arg_loader)
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    #Initial logging
    logging.info(f'''Starting training:
        Epochs:          {args.total_epoch}
        warm up epoch:   {args.warmup_epoch}
        Batch size:      {args.batch_size}
        Loss metirc      {args.loss}
        Training size:   {len(dataset)}
        checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    net.to(device)
    teacher.to(device)
    initial_teacher_state = deepcopy(teacher.state_dict())
    set_grad(model=teacher, is_requires_grad=False)
    loss_fn = Distribution_loss()
    loss_fn.set_metric(args.loss)
    #begin to train model
    epoch_losses = []
    for i in range(1, args.total_epoch+1):
        net.train()
        teacher.train()
        epoch_loss = 0
        # adjust the learning rate
        lr = cosine_decay_with_warmup(current_iter=i,total_iter=args.total_epoch,warmup_iter=args.warmup_epoch,base_lr=args.init_lr)
        adjust_lr(optimizer,lr)

        for imgs, style_imgs in tqdm(train_loader):

            imgs = imgs.to(dtype=torch.float32,device = device)
            style_imgs = style_imgs.to(dtype=torch.float32, device = device)
            logit_s = net(style_imgs)
            logit_t, _, _ = teacher(imgs,style_imgs)
            logit_t = logit_t.detach()
            loss = loss_fn(logit_t, logit_s)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # update teacher model and it also apply the situation which  both architectures of student model and teacher model are different
        with torch.no_grad():
            m = args.momentum  # momentum parameter
            student_name_parameters = { param[0]:param[1].data.detach() for param in net.named_parameters()}
            for param_t in teacher.named_parameters():
                if param_t[0] in student_name_parameters.keys() and (param_t[0].split(".")[0] != "encoder"): # 不更新teacher model 中 encoder中的參數
                    param_t[1].data = param_t[1].data.mul_(m).add_((1-m)*student_name_parameters[param_t[0]])

        logging.info(f'Training loss: {epoch_loss:6.4f} at epoch {i}.')
        epoch_losses.append(epoch_loss)

        if (save_checkpoint) :
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,f'student_{i}.pth'))
            torch.save(teacher.state_dict(), os.path.join(dir_checkpoint,f'teacher_{i}.pth'))
            logging.info(f'Model saved at epoch {i}.')
        
    min_loss_at = torch.argmin(torch.tensor(epoch_losses)).item() + 1 
    logging.info(f'min Training loss at epoch {min_loss_at}.')
            
    return

class Distribution_loss(torch.nn.Module):
    """p is target distribution and q is predict distribution"""
    def __init__(self):
        super(Distribution_loss, self).__init__()
        self.metric = self.set_metric()

    def kl_divergence(self,p,q):
        """p and q are both a logit(before softmax function)"""
        prob_p = torch.softmax(p,dim=1)
        kl = (prob_p * torch.log_softmax(p,dim=1)) - (prob_p * torch.log_softmax(q,dim=1))
        # print(f"p*torch.log(p) is {torch.sum(p*torch.log(p))}")
        # print(f"p*torch.log(q) is {torch.sum(p*torch.log(q))}")
        # print(f"mean kl divergence: {torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])}")
        return torch.sum(kl) / (kl.shape[0]*kl.shape[-1]*kl.shape[-2])

    def cross_entropy(self,p,q):
        """p and q are both a logit(before softmax function)""" 
        ce = -torch.softmax(p, dim=1) * torch.log_softmax(q, dim=1)
        # print(f"mean ce: {torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])}")
        return torch.sum(ce) / (ce.shape[0]*ce.shape[-1]*ce.shape[-2])
    
    def dice_loss(self,p,q):
        smooth = 1e-8
        prob_p = torch.softmax(p,dim=1)
        prob_q = torch.softmax(q,dim=1)

        inter = torch.sum(prob_p*prob_q) + smooth
        union = torch.sum(prob_p) + torch.sum(prob_q) + smooth
        loss = 1 - ((2*inter) / union)
        return  loss / p.size(0) # loss除以batch size

    def forward(self,p,q):
        assert p.dim() == 4, f"dimension of target distribution has to be 4, but get {p.dim()}"
        assert p.dim() == q.dim(), f"dimension dismatch between p and q"
        if self.metric == 'kl_divergence':
            return self.kl_divergence(p,q)
        elif self.metric == "cross_entropy":
            return self.cross_entropy(p,q)
        elif self.metric == "dice_loss":
            return self.dice_loss(p,q)
        else:
            raise NotImplementedError("the loss metric has not implemented")
        
    def set_metric(self, metric:str="cross_entropy"):
        if metric in ["kl_divergence", "cross_entropy", "dice_loss"]:
            self.metric = metric
        else:
            raise NotImplementedError(f"the loss metric has not implemented. metric name must be in kl_divergence or cross_entropy")

if __name__ == '__main__':
    main()