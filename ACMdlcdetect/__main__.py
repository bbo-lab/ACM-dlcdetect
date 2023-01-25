import argparse
import os
import sys

from ACMdlcdetect.helpers import config_from_module

def main():
    # Parse inputs
    parser = argparse.ArgumentParser(
        description="ACM-DLCdetect - Wrapper for DLC to support label files and CCV videos used bz MPINB-BBOs ACM framework.")
    parser.add_argument('INPUT_PATH', type=str, help="Directory with job configuration")
    parser.add_argument("--headless", action="store_true",
                        help="Run headless")
    parser.add_argument("--nomain", action="store_true",
                        help="Do not run preparation and training")
    args = parser.parse_args()

    if args.headless:
        os.environ['DLClight'] = 'True'

    # Load config
    # TODO change config system, e.g. pass around a dictionary instead of importing the config everywhere, requiring the sys.path.insert
    input_path = os.path.expanduser(args.INPUT_PATH)
    sys.path.insert(0, input_path)
    print(f'Loading {input_path} ...')
    import dlcdetectConfig as cfg
    project_name = os.path.basename(os.path.dirname(cfg.__file__))
    config = config_from_module(cfg)

    if not args.nomain:
        from ACMdlcdetect import main
        main.main(project_name, config)

    from ACMdlcdetect import save
    save.main(project_name, config)


if __name__ == '__main__':
    main()
