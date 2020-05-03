import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # all arguments go here;
    parser.add_argument('--root_dir', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Images-processed', help='root director to the data')
    parser.add_argument('--train_npy', type=str, default='./train_list.npy', help='root director to the data')
    parser.add_argument('--test_npy', type=str, default='./test_list.npy', help='root director to the data')
    parser.add_argument('--val_npy', type=str, default='./val_list.npy', help='root director to the data')
    
    args= parser.parse_args()
    return args
