import argparse
import os
import sys
from pprint import pprint

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description="ACM-DLCdetect - Wrapper for DLC to support label files and CCV videos used bz MPINB-BBOs ACM framework.")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with job configuration")
    parser.add_argument("--headless", action="store_true",
                    help="Run headless")
    args = parser.parse_args()
    input_path = os.path.expanduser(args.INPUT_PATH)

    # Load config
    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert
    sys.path.insert(0,input_path)
    print(f'Loading {input_path} ...')

    if args.headless:
        os.environ['DLClight'] = 'True'

    from . import main
    from . import save

    main.main()
    save.main()

if __name__ == '__main__':
    main()
