"""
Calibration data loading utilities.

Provides functions for locating, loading, and saving calibration JSON files.
"""

from pathlib import Path
from typing import Optional

from tools.calibration_schema import CalibrationData


# Directory where calibration files are stored
CALIBRATIONS_DIR = Path(__file__).parent


def get_calibration_path(game_slug: str, schema_name: str) -> Path:
    """
    Get the path to a calibration file.

    Args:
        game_slug: Game identifier (e.g., 'dragon_age_inquisition')
        schema_name: Schema name (e.g., 'dai_human_female')

    Returns:
        Path to the calibration JSON file
    """
    return CALIBRATIONS_DIR / game_slug / f"{schema_name}.json"


def calibration_exists(game_slug: str, schema_name: str) -> bool:
    """Check if a calibration file exists."""
    return get_calibration_path(game_slug, schema_name).exists()


def load_calibration(game_slug: str, schema_name: str) -> Optional[CalibrationData]:
    """
    Load calibration data for a schema.

    Args:
        game_slug: Game identifier
        schema_name: Schema name

    Returns:
        CalibrationData if file exists, None otherwise
    """
    path = get_calibration_path(game_slug, schema_name)
    if not path.exists():
        return None
    return CalibrationData.load(path)


def save_calibration(calibration: CalibrationData, game_slug: str, schema_name: str) -> Path:
    """
    Save calibration data to the appropriate path.

    Creates the game directory if it doesn't exist.

    Args:
        calibration: The calibration data to save
        game_slug: Game identifier
        schema_name: Schema name

    Returns:
        Path where the file was saved
    """
    path = get_calibration_path(game_slug, schema_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    calibration.save(path)
    return path


def list_calibrations(game_slug: Optional[str] = None) -> list[tuple[str, str]]:
    """
    List available calibrations.

    Args:
        game_slug: If provided, only list calibrations for this game

    Returns:
        List of (game_slug, schema_name) tuples
    """
    calibrations = []

    if game_slug:
        game_dir = CALIBRATIONS_DIR / game_slug
        if game_dir.exists():
            for json_file in game_dir.glob("*.json"):
                calibrations.append((game_slug, json_file.stem))
    else:
        for game_dir in CALIBRATIONS_DIR.iterdir():
            if game_dir.is_dir() and not game_dir.name.startswith('_'):
                for json_file in game_dir.glob("*.json"):
                    calibrations.append((game_dir.name, json_file.stem))

    return sorted(calibrations)
