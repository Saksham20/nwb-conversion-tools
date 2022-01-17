"""Authors: Cody Baker, Alessio Buccino."""
import re
import numpy as np
from pathlib import Path
from importlib import import_module
from itertools import chain
from jsonschema import validate, RefResolver
from collections import OrderedDict

from .json_schema import dict_deep_update, load_dict_from_file, FilePathType, OptionalFolderPathType
from ..nwbconverter import NWBConverter


def check_regular_timestamps(ts):
    """Check whether rate should be used instead of timestamps."""
    time_tol_decimals = 9
    uniq_diff_ts = np.unique(np.diff(ts).round(decimals=time_tol_decimals))
    return len(uniq_diff_ts) == 1


def run_conversion_from_yaml(
    specification_file_path: FilePathType,
    data_folder: OptionalFolderPathType = None,
    output_folder: OptionalFolderPathType = None,
    overwrite: bool = False,
):
    """
    Run conversion to NWB given a yaml specification file.

    Parameters
    ----------
    specification_file_path : FilePathType
        File path leading to .yml specification file for NWB conversion.
    data_folder : FolderPathType, optional
        Folder path leading to root location of the data files.
        The default is the parent directory of the specification_file_path.
    output_folder : FolderPathType, optional
        Folder path leading to the desired output location of the .nwb files.
        The default is the parent directory of the specification_file_path.
    overwrite : bool, optional
        If True, replaces any existing NWBFile at the nwbfile_path location, if save_to_file is True.
        If False, appends the existing NWBFile at the nwbfile_path location, if save_to_file is True.
        The default is False.
    """
    if data_folder is None:
        data_folder = Path(specification_file_path).parent
    if output_folder is None:
        output_folder = Path(specification_file_path).parent

    specification = load_dict_from_file(file_path=specification_file_path)
    schema_folder = Path(__file__).parent.parent / "schemas"
    specification_schema = load_dict_from_file(file_path=schema_folder / "yaml_conversion_specification_schema.json")
    validate(
        instance=specification,
        schema=specification_schema,
        resolver=RefResolver(base_uri="file://" + str(schema_folder) + "/", referrer=specification_schema),
    )

    global_metadata = specification.get("metadata", dict())
    global_data_interfaces = specification.get("data_interfaces")
    nwb_conversion_tools = import_module(
        name=".",
        package="nwb_conversion_tools",  # relative import, but named and referenced as if it were absolute
    )
    for experiment in specification["experiments"].values():
        experiment_metadata = experiment.get("metadata", dict())
        experiment_data_interfaces = experiment.get("data_interfaces")
        for session in experiment["sessions"]:
            session_data_interfaces = session.get("data_interfaces")
            data_interface_classes = dict()
            data_interfaces_names_chain = chain(
                *[
                    data_interfaces
                    for data_interfaces in [global_data_interfaces, experiment_data_interfaces, session_data_interfaces]
                    if data_interfaces is not None
                ]
            )
            for data_interface_name in data_interfaces_names_chain:
                data_interface_classes.update({data_interface_name: getattr(nwb_conversion_tools, data_interface_name)})

            CustomNWBConverter = type(
                "CustomNWBConverter", (NWBConverter,), dict(data_interface_classes=data_interface_classes)
            )

            source_data = session["source_data"]
            for interface_name, interface_source_data in session["source_data"].items():
                for key, value in interface_source_data.items():
                    source_data[interface_name].update({key: str(Path(data_folder) / value)})

            converter = CustomNWBConverter(source_data=source_data)
            metadata = converter.get_metadata()
            for metadata_source in [global_metadata, experiment_metadata, session.get("metadata", dict())]:
                metadata = dict_deep_update(metadata, metadata_source)

            print(metadata["NWBFile"])
            if "nwbfile_name" not in session:
                # 'subject_id' is required by schema validation if the 'Subject' is specified in metadata
                assert "Subject" in metadata and "session_start_time" in metadata.get("NWBFile"), (
                    "If not specifying an explicit name for the NWBFile ('nwbfile_name'), then both "
                    "metadata['Subject']['subject_id'] and metadata['NWBFile']['session_start_time'] are required!"
                )
                subject_file_name = metadata["Subject"]["subject_id"].replace(" ", "_")
                nwbfile_name = f"{subject_file_name}_{metadata['NWBFile']['session_start_time']}"
            else:
                nwbfile_name = session["nwbfile_name"]
            converter.run_conversion(
                nwbfile_path=Path(output_folder) / f"{nwbfile_name}.nwb",
                metadata=metadata,
                overwrite=overwrite,
                conversion_options=session.get("conversion_options", dict()),
            )


def reverse_fstring_path(string: str):
    keys = set(re.findall(pattern="\\{(.*?)\\}", string=string))

    adjusted_idx = 0
    if string[0] != "/":
        adjusted_string = "/" + string
        adjusted_idx += 1
    else:
        adjusted_string = string
    if adjusted_string[-1] != "/":
        adjusted_string = adjusted_string + "/"

    sub_paths = adjusted_string.split("/")
    output = dict()
    for key in keys:
        sub_levels = []
        for j, sub_path in enumerate(sub_paths, start=-1):
            if key in sub_path:
                sub_levels.append(j)
        output[key] = sub_levels
    return output


def collect_reverse_fstring_files(string: str):
    adjusted_idx = 0
    if string[0] != "/":
        adjusted_string = "/" + string
        adjusted_idx += 1
    else:
        adjusted_string = string
    if adjusted_string[-1] != "/":
        adjusted_string = adjusted_string + "/"

    sub_paths = adjusted_string.split("/")

    output = reverse_fstring_path(string=string)
    min_level = min(min((values) for values in output.values()))

    # Assumes level to iterate is first occurence of each f-key
    iteration_levels = {key: values[0] for key, values in output.items()}
    inverted_iteration_levels = {value: key for key, value in iteration_levels.items()}
    sorted_iteration_levels = OrderedDict()
    for sorted_value in sorted(iteration_levels.values()):
        sorted_iteration_levels.update({sorted_value - min_level: inverted_iteration_levels[sorted_value]})

    def recur_sub_levels(
        folder_paths,
        n_levels,
        sorted_iteration_levels,
        path,
        level=0,
    ):
        if level < n_levels:
            next_paths = [x for x in path.iterdir()]
            if level == n_levels - 1:
                for next_path in next_paths:
                    path_split = str(next_path).split("/")
                    output = dict(path=next_path)
                    output.update(
                        {
                            fkey: path_split[-(n_levels - fkey_level)]
                            for fkey_level, fkey in sorted_iteration_levels.items()
                        }
                    )
                    folder_paths.append(output)
            else:
                for next_path in next_paths:
                    recur_sub_levels(
                        folder_paths=folder_paths,
                        level=level + 1,
                        n_levels=n_levels,
                        sorted_iteration_levels=sorted_iteration_levels,
                        path=next_path,
                    )

    folder_paths = []
    recur_sub_levels(
        folder_paths=folder_paths,
        n_levels=len(sorted_iteration_levels),
        sorted_iteration_levels=sorted_iteration_levels,
        path=Path("/".join(sub_paths[: min_level + 1])),
    )
    return folder_paths
