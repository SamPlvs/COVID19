import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # all arguments go here;
    parser.add_argument('--batch_size', type=int, default=16, help='the size of data to load onto memory')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--root_dir', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Images-processed', help='root director to the data')
    parser.add_argument('--train_npy', type=str, default='/home/mukund/Documents/COVID/train_list.npy', help='root director to the data')
    parser.add_argument('--test_npy', type=str, default='/home/mukund/Documents/COVID/val_listX.npy', help='root director to the data')
    parser.add_argument('--val_npy', type=str, default='/home/mukund/Documents/COVID/val_list.npy', help='root director to the data')

    parser.add_argument('--train', action='store_true', help='call for training')
    parser.add_argument('--test', action='store_true', help='call for testing')
    parser.add_argument('--cuda', action='store_true', help ='call to use GPU')

    parser.add_argument('--eval_interval', type=int, default=2, help='intervals at which evaluation will take place.')
    parser.add_argument('--save_interval', type=int, default=5, help='intervals at which you aim to save the model')
    args= parser.parse_args()
    return args