from typing import Dict, Any

from src.features.features_base import AnnotationHandler, OverallFeature
from src.features.features_maps import annotation_handlers_map, overall_features_map


class FeaturesMapper:
    @staticmethod
    def get_annotation_handler(name: str, params: Dict[str, Any]) -> AnnotationHandler:
        return annotation_handlers_map[name](**params)

    @staticmethod
    def get_overall_feature(name: str, params: Dict[str, Any]) -> OverallFeature:
        return overall_features_map[name](**params)
