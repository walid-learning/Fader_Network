import numpy as np
from evaluation import Evaluation 
import argparse
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))[:-3] + "\\cfg")
from config import PATH




parser = argparse.ArgumentParser()


parser.add_argument("--batch_test", default=32, type=int, help="Batch size used to train the model and to dump images/attributes from dataset")
parser.add_argument("--attributs", default=["Male"], nargs="+", help="Name attributs")
parser.add_argument("--weights", default=f"{PATH}/utils/models/" , type=str, nargs="+", help="Path to model already saved (Discriminator/AutoEncoder)" )

#C:/Users/33660/Desktop/Etudes/Master_2/SEMESTRE_1/MACHINE_LEARNING_AVANCEE/Projet/Fader_Network/utils/models/Model_Smiling/
#parser.add_argument("--weights", default=" " , type=str, nargs="+", help="Path to model already saved (Discriminator/AutoEncoder)" )


args = parser.parse_args()

E = Evaluation(attributs=args.attributs, batch_test=args.batch_test, weights=args.weights)
imgs, x_reconstruct = E.test()
E.plot_testImg()