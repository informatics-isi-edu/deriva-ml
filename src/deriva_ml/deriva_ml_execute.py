from sympy import cxxcode

from  deriva_ml import DerivaML, execution_configuration

def execute(host, catalog, script):
    workflow_rid = foobar
    execution_configuration = cxxcode(

    )
    ml_instance = DerivaML()
    ml_instance.create_execution(configuratio)
    script


from deriva_ml import DerivaML, ExecutionConfiguration, DatasetSpec, RID, DerivaMLException
import os
import sys
import json
import traceback
import argparse
import requests
from requests.exceptions import HTTPError, ConnectionError
from deriva.transfer import GenericDownloader
from deriva.transfer.download import DerivaDownloadError, DerivaDownloadConfigurationError, \
    DerivaDownloadAuthenticationError, DerivaDownloadAuthorizationError, DerivaDownloadTimeoutError, \
    DerivaDownloadBaggingError
from deriva.core import BaseCLI, KeyValuePairArgs, format_credential, format_exception, urlparse


class DerivaMLExecCLI(BaseCLI):
    def __init__(self, description, epilog, **kwargs):

        BaseCLI.__init__(self, description, epilog, **kwargs)
        self.parser.add_argument("--catalog", default=1, metavar="<1>", help="Catalog number. Default: 1")
        self.parser.add_argument("--timeout", metavar="<seconds>",
                                 help="Total number of seconds elapsed before the download is aborted.")
        self.parser.add_argument("output_dir", metavar="<output dir>", help="Path to an output directory.")
        self.parser.add_argument("envars", metavar="[key=value key=value ...]",
                                 nargs=argparse.REMAINDER, action=KeyValuePairArgs, default={},
                                 help="Variable length of whitespace-delimited key=value pair arguments used for "
                                      "string interpolation in specific parts of the configuration file. "
                                      "For example: key1=value1 key2=value2")

    def main(self):
        try:
            args = self.parse_cli()
        except ValueError as e:
            sys.stderr.write(str(e))
            return 2
        if not args.quiet:
            sys.stderr.write("\n")

        try:
            try:
                ml_instance = DerivaML(args.hostname, args.catalog)
                downloaded = self.execute()
                sys.stdout.write("\n%s\n" % (json.dumps(downloaded)))
            except ConnectionError as e:
                raise DerivaDownloadError("Connection error occurred. %s" % format_exception(e))
            except HTTPError as e:
                if e.response.status_code == requests.codes.unauthorized:
                    raise DerivaDownloadAuthenticationError(
                        "The requested service requires authentication and a valid login session could "
                        "not be found for the specified host. Server responded: %s" % e)
                elif e.response.status_code == requests.codes.forbidden:
                    raise DerivaDownloadAuthorizationError(
                        "A requested operation was forbidden. Server responded: %s" % e)
        except (DerivaDownloadError, DerivaDownloadConfigurationError, DerivaDownloadAuthenticationError,
                DerivaDownloadAuthorizationError, DerivaDownloadTimeoutError, DerivaDownloadBaggingError) as e:
            sys.stderr.write(("\n" if not args.quiet else "") + format_exception(e))
            if args.debug:
                traceback.print_exc()
            return 1
        except:
            sys.stderr.write("An unexpected error occurred.")
            traceback.print_exc()
            return 1
        finally:
            if not args.quiet:
                sys.stderr.write("\n\n")
        return 0


def do_stuff():
    pass

def main(datasets: list[RID], model: list[RID], hostname: str, catalog_id: str):
    my_url = DerivaML.github_url()
    ml_instance = DerivaML(hostname, catalog_id)
    ml_instance.lookup_workflow(my_url)
    config = ExecutionConfiguration(
        datasets=[DatasetSpec(rid=dataset,
                              version=ml_instance.dataset_version(dataset)) for dataset in datasets],
        assets=model,
        workflow= ml_instance.lookup_workflow(my_url)
    )
    execution = ml_instance.create_execution(config)
    with execution as e:
        do_stuff()
    execution.upload_execution_outputs()

if __name__ == "__main__":
    main(datasets, model, hostname, catalog_id)
if __file__ == matplotlib_inline
