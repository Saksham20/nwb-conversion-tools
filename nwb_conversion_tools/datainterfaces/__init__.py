from .neuroscopedatainterface import (
    NeuroscopeRecordingInterface,
    NeuroscopeLFPInterface,
    NeuroscopeMultiRecordingTimeInterface,
    NeuroscopeSortingInterface
)
from .spikeglxdatainterface import SpikeGLXRecordingInterface, SpikeGLXLFPInterface
from .sipickledatainterfaces import (
    SIPickleRecordingExtractorInterface,
    SIPickleSortingExtractorInterface)
from .intandatainterface import IntanRecordingInterface
from .ceddatainterface import CEDRecordingInterface
from .cellexplorerdatainterface import CellExplorerSortingInterface
from .roiextractordatainterface import (
    CaimanSegmentationInterface,
    CnmfeSegmentationInterface,
    Suite2pSegmentationInterface,
    ExtractSegmentationInterface,
    SimaSegmentationInterface
)
from .imagingextractorinterface import (
    SbxImagingInterface,
    TiffImagingInterface,
    Hdf5ImagingInterface
)

from .giocomoconverter import GiocomoImagingInterface
interface_list = [
    NeuroscopeRecordingInterface,
    NeuroscopeSortingInterface,
    SpikeGLXRecordingInterface,
    SIPickleRecordingExtractorInterface,
    SIPickleSortingExtractorInterface,
    IntanRecordingInterface,
    CellExplorerSortingInterface,
    CEDRecordingInterface,
    CaimanSegmentationInterface,
    CnmfeSegmentationInterface,
    Suite2pSegmentationInterface,
    ExtractSegmentationInterface,
    SimaSegmentationInterface,
    SbxImagingInterface,
    TiffImagingInterface,
    Hdf5ImagingInterface,
    GiocomoImagingInterface
]