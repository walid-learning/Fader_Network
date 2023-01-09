import numpy as np
from train import Train 
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import PATH



parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=0.1, help="learning rate")
parser.add_argument("--epochs", default=3, type=int, help="number of epochs to train model")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size used to train the model and to dump images/attributes from dataset")
parser.add_argument("--attributs", default=["Male"], nargs="+", help="Name attributs")
parser.add_argument("--nbr_itr", default=8, type=int, help="number of iteration per epoch")
parser.add_argument("--weights", default=" " , type=str, nargs="+", help="Path to model already saved (Discriminator/AutoEncoder)" )

# f"{PATH}/utils/models/"
#C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/MACHINE_LEARNING_AVANCEE/Projet/Fader_Network/utils/models/Model_Smiling/
#parser.add_argument("--weights", default=" " , type=str, nargs="+", help="Path to model already saved (Discriminator/AutoEncoder)" )


args = parser.parse_args()

T = Train(lr = args.learning_rate, attributs=args.attributs, epochs=args.epochs, nbr_itr_epoch=args.nbr_itr)
T.training(batch_size = args.batch_size, weights=args.weights)
T.plot_loss()

