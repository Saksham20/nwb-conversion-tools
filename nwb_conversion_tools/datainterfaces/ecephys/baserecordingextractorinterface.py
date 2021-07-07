"""Authors: Cody Baker and Ben Dichter."""
from abc import ABC
from typing import Union, Optional
from pathlib import Path
import numpy as np

import spikeextractors as se
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, ElectricalSeries

from ...basedatainterface import BaseDataInterface
from ...utils.json_schema import (
    get_schema_from_hdmf_class,
    get_schema_from_method_signature,
    fill_defaults,
    get_base_schema
)
from ...utils.spike_interface import write_recording

OptionalPathType = Optional[Union[str, Path]]


class BaseRecordingExtractorInterface(BaseDataInterface, ABC):
    """Primary class for all RecordingExtractorInterfaces."""

    RX = None

    @classmethod
    def get_source_schema(cls):
        """Compile input schema for the RecordingExtractor."""
        return get_schema_from_method_signature(cls.RX.__init__)

    def __init__(self, **source_data):
        super().__init__(**source_data)
        self.recording_extractor = self.RX(**source_data)
        self.subset_channels = None
        self.source_data = source_data

    def get_metadata_schema(self):
        """Compile metadata schema for the RecordingExtractor."""
        metadata_schema = super().get_metadata_schema()

        # Initiate Ecephys metadata
        metadata_schema['properties']['Ecephys'] = get_base_schema(tag='Ecephys')
        metadata_schema['properties']['Ecephys']['required'] = ['Device', 'ElectrodeGroup']
        metadata_schema['properties']['Ecephys']['properties'] = dict(
            Device=dict(
                type="array",
                minItems=1,
                items={"$ref": "#/properties/Ecephys/properties/definitions/Device"}
            ),
            ElectrodeGroup=dict(
                type="array",
                minItems=1,
                items={"$ref": "#/properties/Ecephys/properties/definitions/ElectrodeGroup"}
            ),
            Electrodes=dict(
                type="array",
                minItems=0,
                renderForm=False,
                items={"$ref": "#/properties/Ecephys/properties/definitions/Electrodes"}
            ),
        )
        # Schema definition for arrays
        metadata_schema['properties']['Ecephys']['properties']["definitions"] = dict(
            Device=get_schema_from_hdmf_class(Device),
            ElectrodeGroup=get_schema_from_hdmf_class(ElectrodeGroup),
            Electrodes=dict(
                type="object",
                additionalProperties=False,
                required=["name"],
                properties=dict(
                    name=dict(
                        type="string",
                        description="name of this electrodes column"
                    ),
                    description=dict(
                        type="string",
                        description="description of this electrodes column"
                    )
                )
            )
        )
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['Ecephys'] = dict(
            Device=[
                dict(
                    name='Device_ecephys',
                    description='no description'
                )
            ],
            ElectrodeGroup=[
                dict(
                    name=str(group_id),
                    description="no description",
                    location="unknown",
                    device='Device_ecephys'
                )
                for group_id in np.unique(self.recording_extractor.get_channel_groups())
            ],
        )
        return metadata

    def subset_recording(self, stub_test: bool = False):
        """
        Subset a recording extractor according to stub and channel subset options.

        Parameters
        ----------
        stub_test : bool, optional (default False)
        """
        kwargs = dict()

        if stub_test:
            num_frames = 100
            end_frame = min([num_frames, self.recording_extractor.get_num_frames()])
            kwargs.update(end_frame=end_frame)

        if self.subset_channels is not None:
            kwargs.update(channel_ids=self.subset_channels)

        recording_extractor = se.SubRecordingExtractor(
            self.recording_extractor,
            **kwargs
        )
        return recording_extractor

    def run_conversion(
        self,
        nwbfile: NWBFile,
        metadata: dict = None,
        stub_test: bool = False,
        use_times: bool = False,
        save_path: OptionalPathType = None,
        overwrite: bool = False,
        buffer_mb: int = 500,
        write_as: str = 'raw',
        es_key: str = None,
    ):
        """
        Primary function for converting raw (unprocessed) recording extractor data to nwb.

        Parameters
        ----------
        nwbfile: NWBFile
            nwb file to which the recording information is to be added
        metadata: dict
            metadata info for constructing the nwb file (optional).
            Should be of the format
                metadata['Ecephys']['ElectricalSeries'] = dict(name=my_name, description=my_description)
        use_times: bool
            If True, the times are saved to the nwb file using recording.frame_to_time(). If False (default),
            the sampling rate is used.
        save_path: PathType
            Required if an nwbfile is not passed. Must be the path to the nwbfile
            being appended, otherwise one is created and written.
        overwrite: bool
            If using save_path, whether or not to overwrite the NWBFile if it already exists.
        stub_test: bool, optional (default False)
            If True, will truncate the data to run the conversion faster and take up less memory.
        buffer_mb: int (optional, defaults to 500MB)
            Maximum amount of memory (in MB) to use per iteration of the internal DataChunkIterator.
            Requires trace data in the RecordingExtractor to be a memmap object.
        write_as: str (optional, defaults to 'raw')
            Options: 'raw', 'lfp' or 'processed'
        es_key: str (optional)
            Key in metadata dictionary containing metadata info for the specific electrical series
        """
        if stub_test or self.subset_channels is not None:
            recording = self.subset_recording(stub_test=stub_test)
        else:
            recording = self.recording_extractor

        write_recording(
            recording=recording,
            nwbfile=nwbfile,
            metadata=metadata,
            use_times=use_times,
            write_as=write_as,
            es_key=es_key,
            save_path=save_path,
            overwrite=overwrite,
            buffer_mb=buffer_mb
        )