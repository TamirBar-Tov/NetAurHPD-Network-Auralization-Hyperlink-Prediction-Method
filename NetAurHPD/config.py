import argparse
import sys

def parse():
    parser = argparse.ArgumentParser()
    
    # Negative sampling
    parser.add_argument('-alpha', type=float, default=0.5)
    parser.add_argument('-beta', type=float, default=1)
    #------------------------------------------------------------------------------------------------------------------------------------

    # network_auralization
    parser.add_argument('--l', type=int, default=10000) # signal length
    # ------------------------------------------------------------------------------------------------------------------------------------

    # train and test split
    parser.add_argument('-train_size', type=float, default=0.6)
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # M5 model
    parser.add_argument('-stride', type=int, default=8)
    parser.add_argument('-n_channel', type=int, default=32)
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    # M5 training
    parser.add_argument('-epochs', type=int, default=70)
    parser.add_argument('-lr', type=int, default=0.001)
    # ------------------------------------------------------------------------------------------------------------------------------------
    
    return parser.parse_args([])