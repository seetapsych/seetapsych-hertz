import numpy as np
from scipy.interpolate import interp1d


class LandmarkMapper:
    """Landmark mapping class - maps 98 landmarks to 81 landmarks."""

    def __init__(self):
        # print("[*] Initializing landmark mapper...")
        pass

    # ==========================================
    # 1. Core mapping algorithm (98 -> 81, piecewise interpolation)
    # ==========================================
    def resample_curve(self, points: np.ndarray, num_target_points: int, kind: str = "linear") -> np.ndarray:
        """
        Uniformly resample (interpolate) a curve based on arc length.
        Fix: when the curve length is close to zero, generate uniformly
        distributed points instead of duplicated points.
        """
        if len(points) < 2:
            if len(points) == 1:
                # For a single point, generate uniformly distributed points
                # along the diagonal direction to avoid overlap.
                result = np.zeros((num_target_points, 2))
                for i in range(num_target_points):
                    offset = (i - (num_target_points - 1) / 2) * 2  # Small offset
                    result[i] = points[0] + offset
                return result
            return np.zeros((num_target_points, 2))
        if len(points) == num_target_points:
            return points.copy()

        diffs = np.diff(points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
        total_length = cumulative_dist[-1]

        if total_length < 1e-6:  # Curve length is close to zero
            # Generate points uniformly distributed along the curve direction.
            direction = points[-1] - points[0]
            if np.linalg.norm(direction) < 1e-6:
                # All points overlap; generate points with diagonal offsets.
                result = np.zeros((num_target_points, 2))
                for i in range(num_target_points):
                    offset = (i - (num_target_points - 1) / 2) * 2
                    result[i] = points[0] + offset
                return result
            direction = direction / np.linalg.norm(direction)
            result = np.zeros((num_target_points, 2))
            for i in range(num_target_points):
                t = i / (num_target_points - 1)
                result[i] = points[0] + t * direction * 5  # Small length of 5 pixels
            return result

        t_normalized = cumulative_dist / total_length
        interpolator = interp1d(t_normalized, points, axis=0, kind=kind)
        t_target = np.linspace(0, 1, num_target_points)

        return np.asarray(interpolator(t_target))

    # ==========================================
    # 2. **Core correction**: nose semantic mapping and geometric estimation
    # ==========================================
    def generate_nose_81_interpolated(self, pts98: np.ndarray) -> np.ndarray:
        """
        Semantic alignment: reconstruct the 12 nose landmarks [35-46]
        shown in Figure 1.
        In the WFLW 98-point layout, the nose contains only the vertical
        structure and the nose base, so the "bilateral nasal bridge"
        needs to be estimated geometrically.
        """
        # [Key geometric anchor definitions]
        n51 = pts98[51]  # Top of the nasal root
        n53 = pts98[53]  # Lower part of the nasal bridge

        n55 = pts98[55]  # Leftmost point of the left ala (Alare Left)
        n56 = pts98[56]  # Middle of the left ala (Nose Tip)
        n57 = pts98[57]  # Center of the nose base (Subnasale)
        n58 = pts98[58]  # Middle of the right ala (Nose Tip)
        n59 = pts98[59]  # Rightmost point of the right ala (Alare Right)

        # Horizontal reference vector used to estimate the width
        # (for example, using the distance between the inner eye corners).
        # Here, the nose-base width between points 55 and 59 is used directly
        # because it provides a relatively stable estimate.
        base_width_vector = n59 - n55

        # Standard facial proportion: the nasal root width is approximately
        # 0.5-0.6 times the nose-base width.
        bridge_width_factor = 0.6

        nose_pts_81 = np.zeros((12, 2))

        # ----------------------------------------------------------------------
        # A. Rigid semantic alignment points
        #    (no interpolation required; physical semantics match directly)

        # Jawline semantic alignment
        # Figure 1 point 36 (center of the nose base):
        # semantically matches 98-point n57 (Subnasale).
        nose_pts_81[36 - 35] = n57

        # Figure 1 point 41 (outermost point of the left ala):
        # semantically matches 98-point n55.
        nose_pts_81[41 - 35] = n55

        # Figure 1 point 42 (outermost point of the right ala):
        # semantically matches 98-point n59.
        nose_pts_81[42 - 35] = n59

        # Figure 1 point 43 (midpoint of the left ala):
        # semantically matches 98-point n56.
        nose_pts_81[43 - 35] = n56

        # Figure 1 point 44 (midpoint of the right ala):
        # semantically matches 98-point n58.
        nose_pts_81[44 - 35] = n58

        # Figure 1 point 35 (middle section of the nasal bridge):
        # semantically matches 98-point n52.
        nose_pts_81[35 - 35] = n53

        # Figure 1 point 45 (left nostril):
        # semantically matches 98-point n56 (estimated).
        nose_pts_81[45 - 35] = n56

        # Figure 1 point 46 (right nostril):
        # semantically matches 98-point n58 (estimated).
        nose_pts_81[46 - 35] = n58

        # ----------------------------------------------------------------------
        # B. Geometrically estimated points
        #    (geometric filling for the "nasal bridge width control points"
        #    missing from the 98-point layout)
        # Keep the nasal bridge shape and physical width as consistent as
        # possible with Figure 1.
        # ----------------------------------------------------------------------

        # Figure 1 point 38 (upper left nasal root),
        # point 37 (upper right nasal root).
        # Semantics: located at the Y-axis height of n51 and expanded horizontally.
        width_high = base_width_vector * bridge_width_factor
        nose_pts_81[38 - 35] = n51 + 0.5 * width_high
        nose_pts_81[37 - 35] = n51 - 0.5 * width_high

        width_high = base_width_vector * bridge_width_factor * 2
        nose_pts_81[39 - 35] = n53 + 0.5 * width_high
        nose_pts_81[40 - 35] = n53 - 0.5 * width_high

        return nose_pts_81

    # ==========================================
    # 3. Overall mapping pipeline (combines all facial regions)
    # ==========================================
    def map_98_to_81(self, landmarks_98: np.ndarray) -> np.ndarray:
        """
        High-precision geometric mapping pipeline from 98 to 81 landmarks,
        with the nose region reconstructed as the core correction.
        """
        pts = np.array(landmarks_98)

        landmarks_81_pred = np.zeros((81, 2))

        # [Eyes and pupils] (rigid alignment) - excellent semantic correspondence

        # Map the left eye
        landmarks_81_pred[0] = pts[96]
        landmarks_81_pred[1] = pts[60]
        landmarks_81_pred[2] = pts[64]
        landmarks_81_pred[3] = pts[62]
        landmarks_81_pred[4] = pts[66]
        landmarks_81_pred[5] = pts[61]
        landmarks_81_pred[6] = pts[67]
        landmarks_81_pred[7] = pts[63]
        landmarks_81_pred[8] = pts[65]

        # Map the right eye
        landmarks_81_pred[9] = pts[97]
        landmarks_81_pred[10] = pts[68]
        landmarks_81_pred[11] = pts[72]
        landmarks_81_pred[12] = pts[70]
        landmarks_81_pred[13] = pts[74]
        landmarks_81_pred[14] = pts[69]
        landmarks_81_pred[15] = pts[75]
        landmarks_81_pred[16] = pts[71]
        landmarks_81_pred[17] = pts[73]

        # [Eyebrows] (dimensionality reduction/resampling 9 -> 8)
        # Eyebrow geometry is consistent.

        # Map the left eyebrow
        landmarks_81_pred[18] = pts[33]
        landmarks_81_pred[19] = pts[38]
        landmarks_81_pred[20] = pts[35]
        landmarks_81_pred[21] = pts[40]
        landmarks_81_pred[22] = pts[34]
        landmarks_81_pred[23] = pts[41]
        landmarks_81_pred[24] = pts[36]
        landmarks_81_pred[25] = pts[39]

        # Map the right eyebrow
        landmarks_81_pred[26] = pts[50]
        landmarks_81_pred[27] = pts[46]
        landmarks_81_pred[28] = pts[44]
        landmarks_81_pred[29] = pts[48]
        landmarks_81_pred[30] = pts[43]
        landmarks_81_pred[31] = pts[49]
        landmarks_81_pred[32] = pts[45]
        landmarks_81_pred[33] = pts[47]

        # [Nose] (core correction: reconstruction of 12 landmarks)
        nose = self.generate_nose_81_interpolated(pts)
        landmarks_81_pred[34:46] = nose

        # [Mouth] (dimensionality reduction/resampling 20 -> 14,
        # with key mouth-corner anchors)
        landmarks_81_pred[46] = pts[76]
        landmarks_81_pred[47] = pts[82]
        landmarks_81_pred[48] = pts[79]
        landmarks_81_pred[49] = pts[90]
        landmarks_81_pred[50] = pts[77]
        landmarks_81_pred[51] = pts[81]
        landmarks_81_pred[52] = pts[89]
        landmarks_81_pred[53] = pts[91]
        landmarks_81_pred[54] = pts[94]
        landmarks_81_pred[55] = pts[85]
        landmarks_81_pred[56] = pts[95]
        landmarks_81_pred[57] = pts[93]
        landmarks_81_pred[58] = (pts[87] + pts[86]) * 0.5
        landmarks_81_pred[59] = (pts[84] + pts[83]) * 0.5

        # [Facial contour] (dimensionality reduction/resampling 33 -> 21)
        face_contour = self.resample_curve(pts[0:33], 21)
        landmarks_81_pred[60] = face_contour[0]
        landmarks_81_pred[61] = face_contour[-1]
        landmarks_81_pred[62] = face_contour[0]
        landmarks_81_pred[63] = face_contour[-1]
        landmarks_81_pred[64] = pts[16]
        landmarks_81_pred[65:73] = face_contour[1:9]
        landmarks_81_pred[73:81] = face_contour[-2:-10:-1]

        assert len(landmarks_81_pred) == 81, (
            f"Incorrect number of generated landmarks: expected 81, got {len(landmarks_81_pred)}"
        )
        return landmarks_81_pred
