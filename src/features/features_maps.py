from src.features.annotation_handlers.line_dist import LineDistanceHandler
from src.features.annotation_handlers.line_edited import LineEditedHandler
from src.features.annotation_handlers.norm_num_edited_lines import NormalizedNumberEditedLinesHandler
from src.features.annotation_handlers.norm_num_edited_lines_window import \
    NormalizedNumberEditedLinesWindowHandler
from src.features.annotation_handlers.num_different_ts import NumberDifferentTsHandler
from src.features.annotation_handlers.time_from_first_edit import TimeFromFirstEditHandler
from src.features.annotation_handlers.weight_last_edit import WeightLastEditHandler
from src.features.annotation_handlers.weight_norm_num_lines_last_edited import \
    WeightedNormalizedNumberLinesLastEditedHandler
from src.features.overall_features.first_edited_frame import FirstEditedFrame
from src.features.overall_features.part_edited_error_lines import PartEditedErrorLines
from src.features.overall_features.part_edited_frames import PartEditedFrames
from src.features.overall_features.part_edited_lines import PartEditedLines
from src.features.overall_features.part_edited_lines_max_idf import PartEditedLinesMaxIdfFrame

annotation_handlers_map = {
    "line_dist": LineDistanceHandler,
    "line_edited": LineEditedHandler,
    "norm_num_edited_lines": NormalizedNumberEditedLinesHandler,
    "norm_num_edited_lines_window": NormalizedNumberEditedLinesWindowHandler,
    "num_different_ts": NumberDifferentTsHandler,
    "weight_last_edit": WeightLastEditHandler,
    "weight_norm_num_lines_last_edited": WeightedNormalizedNumberLinesLastEditedHandler,
    "time_from_first_edit": TimeFromFirstEditHandler
}


overall_features_map = {
    "first_edited_frame": FirstEditedFrame,
    "part_edited_error_lines": PartEditedErrorLines,
    "part_edited_frames": PartEditedFrames,
    "part_edited_lines": PartEditedLines,
    "part_edited_lines_max_idf_frame": PartEditedLinesMaxIdfFrame
}
