import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', type=str, nargs="+", help='foo help')

args = parser.parse_args()

print(args.foo)