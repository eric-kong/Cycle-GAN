import tensorflow as tf 
import argparse

parser = argparse.ArgumentParser(description="Parse Input Arguments")
parser.add_argument("--dataset_dir", dest="dataset_path", default="dataset/horse2zebra", help="path of the dataset")


def main(unused_argv):


if __name__ == "__main__":
	tf.app.run()