
import numpy as np
from scipy.interpolate import interp1d

class LandmarkMapper:
    """Landmark mapper - maps 280 points to 81 points."""

    # ==========================================
    # 1. Core mapping algorithm (280 pts -> 81 pts piecewise interpolation)
    # ==========================================
    def resample_curve(self, points, num_target_points, kind='linear'):
        """
        Uniformly sample (interpolate) a curve based on arc length.
        Fix: when curve length is near zero, generate uniformly distributed
        points instead of duplicate points.
        """
        if len(points) < 2:
            if len(points) == 1:
                # Single point: generate points uniformly distributed along the
                # diagonal direction to avoid overlap.
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

        if total_length < 1e-6:  # Curve length is near zero
            # Generate points uniformly distributed along the curve direction.
            direction = points[-1] - points[0]
            if np.linalg.norm(direction) < 1e-6:
                # All points coincide: generate points offset along the diagonal.
                result = np.zeros((num_target_points, 2))
                for i in range(num_target_points):
                    offset = (i - (num_target_points - 1) / 2) * 2
                    result[i] = points[0] + offset
                return result
            direction = direction / np.linalg.norm(direction)
            result = np.zeros((num_target_points, 2))
            for i in range(num_target_points):
                t = i / (num_target_points - 1)
                result[i] = points[0] + t * direction * 5  # Small 5-pixel length
            return result

        t_normalized = cumulative_dist / total_length
        interpolator = interp1d(t_normalized, points, axis=0, kind=kind)
        t_target = np.linspace(0, 1, num_target_points)

        return interpolator(t_target)

    # ==========================================
    # 2. **Core correction**: nose semantic mapping and geometric estimation
    # ==========================================
    def generate_nose_81_interpolated(self, pts280):
        """
        Semantic alignment: reconstructs the 12 nose landmark points indexed
        [35-46] in Figure 1. WFLW 280-point nose only has vertical and bottom
        points; the lateral nasal bones must be geometrically estimated.
        """
        # [Key geometric anchor definitions]
        n43 = pts280[43]  # Nasion (root of nose bridge)
        n44 = pts280[44]  # Mid nose bridge
        n45 = pts280[45]  # Lower nose bridge
        n46 = pts280[46]  # Pronasale (nose tip)

        n82 = pts280[82]  # Alare Left (leftmost alar)
        n47 = pts280[47]  # Left alar mid (Nose Tip)
        n49 = pts280[49]  # Subnasale (nose base center)
        n51 = pts280[51]  # Right alar mid (Nose Tip)
        n83 = pts280[83]  # Alare Right (rightmost alar)

        n78 = pts280[78]  # Left nose bridge apex
        n79 = pts280[79]  # Right nose bridge apex
        n80 = pts280[80]  # Left lower-mid nose bridge
        n81 = pts280[81]  # Right lower-mid nose bridge

        # Horizontal reference vector for width estimation (e.g. inner canthus
        # spacing). Using nose base width 83-82 directly is more stable.
        base_width_vector = n83 - n82
        # Standard face proportion: nasion width ~= 0.5-0.6x nose base width
        bridge_width_factor = 0.6

        nose_pts_81 = np.zeros((12, 2))

        # ----------------------------------------------------------------------
        # A. Rigid semantic alignment points (no interpolation needed, exact
        #    physical correspondence)

        # Jawline semantic alignment
        # Figure 1 idx 36 (nose base mid): semantic match 280-pt n49 (Subnasale)
        nose_pts_81[36 - 35] = n49
        # Figure 1 idx 41 (leftmost alar): semantic match 280-pt n82
        nose_pts_81[41 - 35] = n82
        # Figure 1 idx 42 (rightmost alar): semantic match 280-pt n83
        nose_pts_81[42 - 35] = n83
        # Figure 1 idx 43 (left alar mid): semantic match 280-pt n47
        nose_pts_81[43 - 35] = n47
        # Figure 1 idx 44 (right alar mid): semantic match 280-pt n51
        nose_pts_81[44 - 35] = n51
        # Figure 1 idx 35 (mid nose bridge): semantic match 280-pt n44
        nose_pts_81[35 - 35] = n44

        # Figure 1 idx 45 (left nostril): semantic match 280-pt n47 (estimated)
        nose_pts_81[45 - 35] = n47
        # Figure 1 idx 46 (right nostril): semantic match 280-pt n51 (estimated)
        nose_pts_81[46 - 35] = n51

        nose_pts_81[37 - 35] = n78
        nose_pts_81[38 - 35] = n79

        nose_pts_81[39 - 35] = n80
        nose_pts_81[40 - 35] = n81

        return nose_pts_81

    # ==========================================
    # 3. Overall mapping pipeline (combines all regions)
    # ==========================================
    def map_280_to_81(self, landmarks_280):
        """
        High-precision 280-to-81 point geometric mapping pipeline with
        nose-region reconstruction as the core correction.
        """
        pts = np.array(landmarks_280)

        landmarks_81_pred = np.zeros((81, 2))

        # [Eye pupils] (rigid alignment) - excellent semantic correspondence
        # Map left eye
        landmarks_81_pred[0] = pts[74]
        landmarks_81_pred[1] = pts[52]
        landmarks_81_pred[2] = pts[55]
        landmarks_81_pred[3] = pts[72]
        landmarks_81_pred[4] = pts[73]
        landmarks_81_pred[5] = pts[53]
        landmarks_81_pred[6] = pts[57]
        landmarks_81_pred[7] = pts[54]
        landmarks_81_pred[8] = pts[56]
        # Map right eye
        landmarks_81_pred[9] = pts[77]
        landmarks_81_pred[10] = pts[58]
        landmarks_81_pred[11] = pts[61]
        landmarks_81_pred[12] = pts[75]
        landmarks_81_pred[13] = pts[76]
        landmarks_81_pred[14] = pts[59]
        landmarks_81_pred[15] = pts[63]
        landmarks_81_pred[16] = pts[60]
        landmarks_81_pred[17] = pts[62]

        # [Eyebrows] (dimension-reduction resampling 9->8) - brow shape preserved
        # Map left brow
        landmarks_81_pred[18] = pts[33]
        landmarks_81_pred[19] = pts[67]
        landmarks_81_pred[20] = pts[35]
        landmarks_81_pred[21] = pts[65]
        landmarks_81_pred[22] = pts[34]
        landmarks_81_pred[23] = pts[64]
        landmarks_81_pred[24] = pts[36]
        landmarks_81_pred[25] = pts[66]
        # Map right brow
        landmarks_81_pred[26] = pts[68]
        landmarks_81_pred[27] = pts[42]
        landmarks_81_pred[28] = pts[40]
        landmarks_81_pred[29] = pts[70]
        landmarks_81_pred[30] = pts[39]
        landmarks_81_pred[31] = pts[69]
        landmarks_81_pred[32] = pts[41]
        landmarks_81_pred[33] = pts[71]

        # [Nose] (core correction: 12-point reconstruction)
        nose = self.generate_nose_81_interpolated(pts)
        landmarks_81_pred[34:46] = nose

        # [Mouth] (dimension-reduction resampling 20->14, key mouth corners anchored)
        landmarks_81_pred[46] = pts[84]
        landmarks_81_pred[47] = pts[90]
        landmarks_81_pred[48] = pts[87]
        landmarks_81_pred[49] = pts[98]
        landmarks_81_pred[50] = (pts[85] + pts[86]) * 0.5
        landmarks_81_pred[51] = (pts[88] + pts[89]) * 0.5
        landmarks_81_pred[52] = pts[97]
        landmarks_81_pred[53] = pts[99]
        landmarks_81_pred[54] = pts[102]
        landmarks_81_pred[55] = pts[93]
        landmarks_81_pred[56] = pts[103]
        landmarks_81_pred[57] = pts[101]
        landmarks_81_pred[58] = (pts[95] + pts[94]) * 0.5
        landmarks_81_pred[59] = (pts[92] + pts[91]) * 0.5

        # [Face contour] (dimension-reduction resampling 33->21)
        face_contour = self.resample_curve(pts[0:33], 21)
        landmarks_81_pred[60] = face_contour[0]
        landmarks_81_pred[61] = face_contour[-1]
        landmarks_81_pred[62] = face_contour[0]
        landmarks_81_pred[63] = face_contour[-1]
        landmarks_81_pred[64] = pts[16]
        landmarks_81_pred[65:73] = face_contour[1:9]
        landmarks_81_pred[73:81] = face_contour[-2:-10:-1]

        assert len(landmarks_81_pred) == 81, f"Point count mismatch: expected 81, got {len(landmarks_81_pred)}"
        return landmarks_81_pred
