"""
Facial landmark extraction and geometric measurement computation.

Extracts 68 dlib landmarks, computes 25 scale-invariant ratios, and
obtains 128-dim face encodings from rendered or real face images.
Used as auxiliary input for the dual-input CodePredictor model.
"""

import numpy as np
import face_recognition
import dlib


# Landmark group indices (dlib 68-point model)
_CHIN = list(range(0, 17))
_LEFT_EYEBROW = list(range(17, 22))
_RIGHT_EYEBROW = list(range(22, 27))
_NOSE_BRIDGE = list(range(27, 31))
_NOSE_TIP = list(range(31, 36))
_LEFT_EYE = list(range(36, 42))
_RIGHT_EYE = list(range(42, 48))
_OUTER_LIP = list(range(48, 60))
_INNER_LIP = list(range(60, 68))

NUM_RATIOS = 25
NUM_ENCODING = 128


def _detect_face(image: np.ndarray):
    """Detect face location, trying HOG first then CNN fallback."""
    locations = face_recognition.face_locations(image)
    if not locations:
        locations = face_recognition.face_locations(image, model='cnn')
    return locations


def extract_landmarks(image: np.ndarray) -> np.ndarray | None:
    """Extract 68 facial landmarks from an image.

    Args:
        image: RGB uint8 image (H, W, 3)

    Returns:
        (68, 2) array of (x, y) landmark coordinates, or None if no face detected.
    """
    locations = _detect_face(image)
    if not locations:
        return None

    landmarks_dict = face_recognition.face_landmarks(image, face_locations=[locations[0]])[0]

    points = []
    for group in ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                   'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']:
        points.extend(landmarks_dict[group])

    # face_recognition returns 72 points (top_lip=12, bottom_lip=12 overlap with outer/inner lip)
    # We need to map to dlib's 68-point ordering
    chin = landmarks_dict['chin']           # 17 pts
    l_brow = landmarks_dict['left_eyebrow']  # 5 pts
    r_brow = landmarks_dict['right_eyebrow'] # 5 pts
    nose_b = landmarks_dict['nose_bridge']   # 4 pts
    nose_t = landmarks_dict['nose_tip']      # 5 pts
    l_eye = landmarks_dict['left_eye']       # 6 pts
    r_eye = landmarks_dict['right_eye']      # 6 pts
    top_lip = landmarks_dict['top_lip']      # 12 pts
    bot_lip = landmarks_dict['bottom_lip']   # 12 pts

    # dlib ordering: chin(17) + brows(10) + nose(9) + eyes(12) + lips(20) = 68
    # outer lip = top_lip[0:7] + bot_lip[0:5] (but reversed for bottom)
    # inner lip = top_lip[7:12] + bot_lip[7:12] (reversed bottom)
    outer_lip = top_lip[:7] + [bot_lip[i] for i in [5, 4, 3, 2, 1]]
    inner_lip = [top_lip[i] for i in [7, 8, 9, 10, 11]] + [bot_lip[i] for i in [9, 8, 7]]

    all_68 = chin + l_brow + r_brow + nose_b + nose_t + l_eye + r_eye + outer_lip + inner_lip
    return np.array(all_68, dtype=np.float32)


def compute_ratios(landmarks: np.ndarray) -> np.ndarray:
    """Compute 25 scale-invariant facial measurement ratios from 68 landmarks.

    All ratios are normalized by face dimensions to be scale-invariant.

    Returns:
        (25,) float32 array of ratios.
    """
    pts = landmarks

    # Face dimensions
    face_width = np.linalg.norm(pts[0] - pts[16])
    chin_to_brow = np.linalg.norm(pts[8] - pts[27])
    face_height = max(chin_to_brow, 1e-6)
    fw = max(face_width, 1e-6)

    def _dist(a, b):
        return np.linalg.norm(pts[a] - pts[b])

    def _midpoint(a, b):
        return (pts[a] + pts[b]) / 2

    # Eye centers
    left_eye_center = pts[_LEFT_EYE].mean(axis=0)
    right_eye_center = pts[_RIGHT_EYE].mean(axis=0)
    inter_eye = np.linalg.norm(left_eye_center - right_eye_center)

    # Eye dimensions
    left_eye_w = _dist(36, 39)
    left_eye_h = (np.linalg.norm(pts[37] - pts[41]) + np.linalg.norm(pts[38] - pts[40])) / 2
    right_eye_w = _dist(42, 45)
    right_eye_h = (np.linalg.norm(pts[43] - pts[47]) + np.linalg.norm(pts[44] - pts[46])) / 2

    # Nose dimensions
    nose_width = _dist(31, 35)
    nose_length = _dist(27, 33)

    # Mouth dimensions
    mouth_width = _dist(48, 54)
    upper_lip_h = (np.linalg.norm(pts[51] - pts[62]) + np.linalg.norm(pts[52] - pts[63])) / 2
    lower_lip_h = (np.linalg.norm(pts[57] - pts[66]) + np.linalg.norm(pts[56] - pts[65])) / 2
    lip_height = upper_lip_h + lower_lip_h

    # Jaw width (at gonion points ~4 and ~12)
    jaw_width = _dist(4, 12)

    # Facial thirds
    brow_center = _midpoint(21, 22)
    eye_center = (left_eye_center + right_eye_center) / 2
    brow_to_eye = np.linalg.norm(brow_center - eye_center)
    eye_to_nose = np.linalg.norm(eye_center - pts[33])
    nose_to_chin = np.linalg.norm(pts[33] - pts[8])

    # Brow positions
    left_brow_height = np.linalg.norm(pts[19] - left_eye_center)
    right_brow_height = np.linalg.norm(pts[24] - right_eye_center)
    avg_brow_height = (left_brow_height + right_brow_height) / 2

    # Chin shape (pointedness)
    chin_width_low = _dist(6, 10)
    chin_pointedness = chin_width_low / fw

    # Forehead estimate (top of brow to face top approximated from face height)
    forehead_height = face_height - nose_to_chin - eye_to_nose

    ratios = np.array([
        face_width / face_height,                     # 0: face aspect ratio
        inter_eye / fw,                                # 1: inter-eye / face width
        nose_width / fw,                               # 2: nose width ratio
        nose_length / face_height,                     # 3: nose length ratio
        mouth_width / fw,                              # 4: mouth width ratio
        lip_height / face_height,                      # 5: lip height ratio
        jaw_width / fw,                                # 6: jaw width / face width (taper)
        left_eye_h / max(left_eye_w, 1e-6),           # 7: left eye shape (h/w)
        right_eye_h / max(right_eye_w, 1e-6),         # 8: right eye shape (h/w)
        brow_to_eye / face_height,                     # 9: brow-eye distance ratio
        eye_to_nose / face_height,                     # 10: eye-nose ratio (middle third)
        nose_to_chin / face_height,                    # 11: nose-chin ratio (lower third)
        chin_pointedness,                              # 12: chin pointedness
        upper_lip_h / max(lower_lip_h, 1e-6),         # 13: upper/lower lip ratio
        avg_brow_height / face_height,                 # 14: brow height ratio
        forehead_height / face_height,                 # 15: forehead ratio
        left_eye_w / fw,                               # 16: left eye width ratio
        right_eye_w / fw,                              # 17: right eye width ratio
        _dist(31, 33) / max(nose_width, 1e-6),        # 18: nostril asymmetry
        _dist(33, 35) / max(nose_width, 1e-6),        # 19: nostril asymmetry R
        _dist(48, 51) / max(mouth_width, 1e-6),       # 20: mouth corner-to-peak L
        _dist(54, 51) / max(mouth_width, 1e-6),       # 21: mouth corner-to-peak R
        _dist(4, 8) / max(_dist(8, 12), 1e-6),        # 22: jaw asymmetry
        _dist(27, 30) / face_height,                   # 23: nose bridge length
        _dist(0, 8) / max(_dist(8, 16), 1e-6),        # 24: face outline asymmetry
    ], dtype=np.float32)

    return ratios


def _get_face_encoder():
    """Lazily load the dlib face encoder model."""
    if not hasattr(_get_face_encoder, '_model'):
        import face_recognition.api as _api
        _get_face_encoder._model = _api.face_encoder
        _get_face_encoder._pose = _api.pose_predictor_68_point
    return _get_face_encoder._model, _get_face_encoder._pose


def get_encoding(image: np.ndarray) -> np.ndarray | None:
    """Get 128-dim face encoding from image.

    Uses dlib directly to avoid face_recognition API compatibility issues.

    Args:
        image: RGB uint8 image (H, W, 3)

    Returns:
        (128,) float32 array, or None if no face detected.
    """
    locations = _detect_face(image)
    if not locations:
        return None

    # Convert face_recognition location (top, right, bottom, left) to dlib rectangle
    top, right, bottom, left = locations[0]
    rect = dlib.rectangle(left, top, right, bottom)

    # Ensure contiguous array for dlib
    img = np.ascontiguousarray(image)

    encoder, pose_predictor = _get_face_encoder()
    shape = pose_predictor(img, rect)

    # Wrap in full_object_detections (plural) for dlib compatibility
    fods = dlib.full_object_detections()
    fods.append(shape)
    descriptors = encoder.compute_face_descriptor(img, faces=fods)
    return np.array(descriptors[0], dtype=np.float32)


def extract_all(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract landmarks, ratios, and encoding in one pass.

    Returns:
        (landmarks_68x2, ratios_25, encoding_128) or None if no face detected.
    """
    landmarks = extract_landmarks(image)
    if landmarks is None:
        return None

    ratios = compute_ratios(landmarks)
    encoding = get_encoding(image)
    if encoding is None:
        encoding = np.zeros(NUM_ENCODING, dtype=np.float32)

    return landmarks, ratios, encoding
