import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # all arguments go here;
    parser.add_argument('--root_dir', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Images-processed', help='root director to the data')
    parser.add_argument('--covid_train_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/COVID/trainCT_COVID.txt', help='root director to the data')
    parser.add_argument('--covid_test_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/COVID/testCT_COVID.txt', help='root director to the data')
    parser.add_argument('--covid_val_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/COVID/valCT_COVID.txt', help='root director to the data')
    parser.add_argument('--healthy_train_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt', help='root director to the data')
    parser.add_argument('--healthy_val_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/NonCOVID/valCT_NonCOVID.txt', help='root director to the data')
    parser.add_argument('--healthy_test_list', type=str, default='/home/mukund/Documents/COVID/COVID-CT/Data-split/NonCOVID/testCT_NonCOVID.txt', help='root director to the data')

    args= parser.parse_args()
    return args