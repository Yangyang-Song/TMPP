import os
import sys
import yaml

proj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')
with open(conf_fp) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Get file directory based on the current machine's nodename
# nodename = os.uname().nodename
nodename ="LAPTOP-UP2D1R34"
file_dir = config['filepath'][nodename]


def main():
    pass


if __name__ == '__main__':
    main()
