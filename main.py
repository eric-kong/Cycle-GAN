import tensorflow as tf 
import argparse
from model import CycleGAN

parser = argparse.ArgumentParser(description="Parse Input Arguments")
parser.add_argument("--dataset", dest="dataset", default="horse2zebra", help="name of the dataset that you want to train (default: horse2zebra)")
parser.add_argument("--image_size", dest="image_size", type=int, default=256, help="the output image size (default: 256)")
parser.add_argument("--batch_size", dest="batch_size", type=int, default=1, help="batch_size (default: 1)")
parser.add_argument("--ngf", dest="ngf", type=int, default=64, help="number of filters for generator (default: 64)")
parser.add_argument("--ndf", dest="ndf", type=int, default=64, help="number of filters for discriminator (default: 64)")
parser.add_argument("--lambda1", dest="lambda1", type=int, default=10, help="importance for cycle-consistency loss (X->Y->X, default:10)")
parser.add_argument("--lambda2", dest="lambda2", type=int, default=10, help="importance for cycle-consistency loss (Y->X->Y, default:10)")
parser.add_argument("--use_lsloss", dest="use_lsloss", type=bool, default=True, help="use least square loss or not (default: True)")
parser.add_argument("--epoches", dest="epoches", type=int, default=200, help="number of epoches to train")



args = parser.parse_args()

def main(unused_argv):
	with tf.Session() as sess:
		print ("==================================")
		print ("[*] Start initializing cyclegan...")
		cyclegan = CycleGAN(sess, "cyclegan", dataset=args.dataset, 
			image_size=args.image_size, batch_size=args.batch_size, 
			ngf=args.ngf, ndf=args.ndf, lambda1=args.lambda1, lambda2=args.lambda2)
		print ("[*] Start training...")
		cyclegan.train(args.epoches, load_model=None, save_freq=0.25)


if __name__ == "__main__":
	tf.app.run()
