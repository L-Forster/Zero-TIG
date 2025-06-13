import os
import sys
import time
import glob
from utils import utils
from PIL import Image
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model.model import *
from dataloader.create_data import CreateDataset
import torch.optim as optim

parser = argparse.ArgumentParser("ZERO-TIG")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=5, help='epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--save', type=str, default='EXP/', help='location to save models')
parser.add_argument('--model_pretrain', type=str, default='weights/BVI-RLV.pt', help='pretrained model')
parser.add_argument('--lowlight_images_path', type=str, default='../data/BVI-RLV/input',
                    help='path for train data')
parser.add_argument('--dpflow_model', type=str, default='dpflow', help='dpflow model')
parser.add_argument('--of_scale', type=int, default=3)
parser.add_argument('--dataset', type=str, default='BVI-RLV', help='dataset')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--accumulation_steps', type=int, default=4, help='gradient accumulation steps')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_device('cuda')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)



    model =Network(args)
    utils.save(model, os.path.join(args.save, 'initial_weights.pt'))
    model.enhance.in_conv.apply(model.enhance_weights_init)
    model.enhance.conv.apply(model.enhance_weights_init)
    model.enhance.out_conv.apply(model.enhance_weights_init)

    try:
        base_weights = torch.load(args.model_pretrain)
        pretrained_dict = base_weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('Loaded pre-trained model from %s.' % args.model_pretrain)
    except:
        logging.info('Model is initialized without pre-trained model.')

    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)

    # Dataset
    TrainDataset = CreateDataset(args, task='train')
    logging.info("Training data: %d", TrainDataset.__len__())
    TestDataset = CreateDataset(args, task='test')
    logging.info("Test data: %d", TestDataset.__len__())

    train_queue = torch.utils.data.DataLoader(
        TrainDataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=args.num_workers, shuffle=False, generator=torch.Generator(device='cuda'))
    test_queue = torch.utils.data.DataLoader(
        TestDataset, batch_size=1,
        pin_memory=True, num_workers=args.num_workers, shuffle=False, generator=torch.Generator(device='cuda'))

    start_epoch = 0
    if args.model_pretrain:
        # ... (model loading remains the same)

    for epoch in range(start_epoch, args.epochs):
        logging.info("train-epoch %03d" % epoch)
        model.train()
        optimizer.zero_grad()
        
        for i, (input) in enumerate(train_queue):
            if args.cuda:
                input = [x.cuda() if isinstance(x, torch.Tensor) else x for x in input]
            
            loss = model._loss(input) / args.accumulation_steps
            loss.backward()
            
            if (i + 1) % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        scheduler.step()

        # Save the model
        if not os.path.exists(os.path.join(args.save, 'model_epochs')):
            os.makedirs(os.path.join(args.save, 'model_epochs'))

        save_path = os.path.join(args.save, 'model_epochs', 'weights_%d.pt' % epoch)
        torch.save(model.state_dict(), save_path)
        logging.info(f"Saved model to {save_path}")

        if epoch % 1 == 0 and i != 0:
            model.eval()
            with torch.no_grad():
                for idx, (input, img_name, img_path, last_img_path) in enumerate(test_queue):
                    model.is_new_seq = utils.sequential_judgment(img_path[0], last_img_path[0])
                    if model.is_new_seq:
                        print("Eval Get this img from: ", img_path, "\n Last img from: ", last_img_path)
                    input = Variable(input, volatile=True).cuda()
                    L_pred1,L_pred2,L2,s2,s21,s22,H2,H11,H12,H13,s13,H14,s14,H3,s3,H3_pred,H4_pred,L_pred1_L_pred2_diff,H13_H14_diff,H2_blur,H3_blur,H3_denoised1,H3_denoised2= model(input)
                    input_name = '%s_%s' % (os.path.basename(os.path.split(img_path[0])[0]), img_name[0])
                    H3_img = save_images(H3)
                    H2_img = save_images(H2)
                    os.makedirs(args.save + '/result/denoise/', exist_ok=True)
                    os.makedirs(args.save + '/result/enhance/', exist_ok=True)
                    Image.fromarray(H3_img).save(args.save + '/result/denoise/' + input_name+'_denoise_'+str(epoch)+'.png', 'PNG')
                    Image.fromarray(H2_img).save(args.save + '/result/enhance/' +input_name+'_enhance_'+str(epoch)+'.png', 'PNG')


if __name__ == '__main__':
    main()