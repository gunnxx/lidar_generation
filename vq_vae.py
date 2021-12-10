import argparse
import torch
import torch.utils.data
import torch.optim as optim
import tensorboardX

from utils import * 
from models import * 

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',     type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--base_dir',       type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',       type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',             type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--embedding_dim',  type=int,   default=128,            help='size of the embedding')
parser.add_argument('--num_embedding',  type=int,   default=256,            help='size of embedding dictionary')
parser.add_argument('--beta',           type=float, default=0.25,           help='weight for part of VQ layer loss')
parser.add_argument('--device',         type=str,   default="cpu",          help='device to be used')
parser.add_argument('--debug', action='store_true')

# ------------------------------------------------------------------------------
args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# construct model and ship to GPU
device = torch.device(args.device)
vq_layer = VectorQuantizer(args).to(device)
model = VQVAE(args, vq_layer).to(device)

# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0
ns     = 16

# dataset preprocessing
print('loading data')
dataset_train = np.load('../lidar_generation/kitti_data/lidar.npz')
dataset_val   = np.load('../lidar_generation/kitti_data/lidar_val.npz') 

if args.debug: 
    dataset_train, dataset_val = dataset_train[:128], dataset_val[:128]

dataset_train = preprocess(dataset_train).astype('float32')
dataset_val   = preprocess(dataset_val).astype('float32')

train_loader  = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, drop_last=True)

val_loader    = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, drop_last=False)

print(model)
model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr) 
loss_fn = F.binary_cross_entropy


# VAE training
# ------------------------------------------------------------------------------

for epoch in range(300):
    print('epoch %s' % epoch)
    model.train()
    loss_, vq_loss_, recon_ = [[] for _ in range(3)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x

    for i, img in enumerate(train_loader):
        img = img.to(device)
        recon, vq_loss = model(process_input(img))

        loss_recon = loss_fn(recon, img)
        loss = loss_recon + vq_loss

        loss_    += [loss.item()]
        vq_loss_ += [vq_loss.item()]
        recon_   += [loss_recon.item()]

        optim.zero_grad()
        loss.backward()
        optim.step()

    writes += 1
    mn = lambda x : np.mean(x)
    print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    print_and_log_scalar(writer, 'train/vq_loss_', mn(vq_loss_), writes)
    print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
        
    # save some training reconstructions
    if epoch % 10 == 0:
         recon = recon[:ns].cpu().data.numpy()
         with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
             np.save(f, recon)

         print('saved training reconstructions')
         
    
    # Testing loop
    # --------------------------------------------------------------------------

    loss_, vq_loss_, recon_ = [[] for _ in range(3)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            print('test set evaluation')
            for i, img in enumerate(val_loader):
                img = img.to(device)
                recon, vq_loss = model(process_input(img))

                loss_recon = loss_fn(recon, img)
                loss = loss_recon + vq_loss

                loss_    += [loss.item()]
                vq_loss_ += [vq_loss.item()]
                recon_   += [loss_recon.item()]

            
            print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
            print_and_log_scalar(writer, 'train/vq_loss_', mn(vq_loss_), writes)
            print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)

            if epoch % 10 == 0:
                with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
                    recon = recon[:ns].cpu().data.numpy()
                    np.save(f, recon)
                    print('saved test recons')
                
            if epoch == 0: 
                with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
                    img = img.cpu().data.numpy()
                    np.save(f, img)
                
                print('saved real LiDAR')

    if (epoch + 1) % 10 == 0 :
        torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
