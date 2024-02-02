import sys
import argparse
import json
from pathlib import Path
from deriva_ml.execution_configuration import ExecutionConfiguration
from deriva_ml.deriva_ml_base import DerivaML
from deriva_ml.execution_template import execution_template


def upload_configuration(config_file: str, hostname: str, catalog_id: str, schema_name: str, desc: str):
    dml = DerivaML(hostname=hostname, catalog_id=catalog_id, schema_name=schema_name)
    dml.upload_execution_configuration(config_file, desc)


def instantiate_template(config_file='execution_configuration.json'):
    # test if the dest file exists, if false, do the copy, or else abort the copy operation.
    path = Path(config_file)
    if not path.exists():
        with open(config_file, "w") as template_file:
            json.dump(execution_template, template_file, indent=4, sort_keys=True)
    else:
        raise FileExistsError


def main():
    parser = argparse.ArgumentParser(
        prog='execution_config',
        description='Validate and upload a DerivaML execution configuration',
        epilog='The Deriva ML library')
    parser.add_argument('-f', '--file',
                        metavar="<path to file>",
                        help="Name of execution configuration file if not execution_configuration.json",
                        default="execution_configuration.json")  # positional argument
    parser.add_argument('-u', '--upload',
                        help="Upload configuration file to <hostname> <catalog id> <schema name> <description",
                        metavar=("<host name>", "<catalog id>", "<schema name>", "<description>"),
                        nargs=4)
    parser.add_argument('-c', '--create',
                        action='store_true',
                        help="Create a new execution configuration")
    args = parser.parse_args()

    if args.create:
        instantiate_template(args.file)
        return
    else:
        with open(args.file, 'r') as file:
            config = json.load(file)
            # check input metadata
            configuration = ExecutionConfiguration.model_validate(config)
    if args.upload:
        upload_configuration(args.file, *args.upload)


if __name__ == '__main__':
    sys.exit(main())
