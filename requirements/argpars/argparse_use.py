import argparse

parser=argparse.ArgumentParser()
parser.add_argument("file")
parser.add_argument("--pp",action="store_true",help="wuwei")
args=parser.parse_args()
print(args.file)
print(args.pp)