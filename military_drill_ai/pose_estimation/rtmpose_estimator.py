import cv2
import numpy as np
import os

try:
    from mmpose.apis import inference_topdown, init_model
except ImportError:
    print("Warning: mmpose is not installed. Please install it to use RTMPose-M.")
    init_model = None

class RTMPoseEstimator:
    def __init__(self, device='cuda:0'):
        """
        Initializes the RTMPose-M model using MMPose.
        Downloads weights and config automatically if not present.
        """
        self.device = device
        
        # RTMPose-M config and weights from OpenMMLab
        self.config_file = 'rtmpose-m_8xb256-420e_coco-256x192.py'
        self.checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth'
        
        # Download config if missing
        if not os.path.exists(self.config_file):
            import urllib.request
            url = 'https://raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
            try:
                urllib.request.urlretrieve(url, self.config_file)
            except Exception as e:
                print(f"Error downloading RTMPose config: {e}")

        if init_model is not None and os.path.exists(self.config_file):
            print("Initializing RTMPose-M model...")
            self.model = init_model(self.config_file, self.checkpoint, device=self.device)
        else:
            self.model = None
            
    def estimate_pose(self, frame, tracks):
        """
        Estimates pose for each tracked person.
        tracks: list of [x1, y1, x2, y2, conf, cls, cadet_id]
        Returns: list of dicts with 'cadet_id', 'keypoints', 'scores'
        """
        if self.model is None or len(tracks) == 0:
            return []
            
        # Extract bboxes in format expected by mmpose: np.array([[x1, y1, x2, y2]])
        bboxes = []
        for t in tracks:
            bboxes.append([t[0], t[1], t[2], t[3]])
            
        bboxes = np.array(bboxes)
        
        # Run top-down inference
        results = inference_topdown(self.model, frame, bboxes, bbox_format='xyxy')
        
        poses = []
        for i, data_sample in enumerate(results):
            pred_instances = data_sample.pred_instances
            
            # RTMPose outputs 17 COCO keypoints per instance
            keypoints = pred_instances.keypoints[0] # (17, 2)
            scores = pred_instances.keypoint_scores[0] # (17,)
            
            poses.append({
                'cadet_id': tracks[i][6],
                'keypoints': keypoints,
                'scores': scores
            })
            
        return poses

    def draw_skeletons(self, frame, poses):
        """
        Draws COCO 17-point skeletons on the frame.
        """
        out_frame = frame.copy()
        
        # COCO 17 keypoint connections
        SKELETON = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
            (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
            (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
            (3, 5), (4, 6)
        ]
        
        for pose in poses:
            kpts = pose['keypoints']
            scores = pose['scores']
            
            # Draw points (Joints)
            for i, (x, y) in enumerate(kpts):
                if scores[i] > 0.4:
                    cv2.circle(out_frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                    
            # Draw lines (Bones)
            for j1, j2 in SKELETON:
                if scores[j1] > 0.4 and scores[j2] > 0.4:
                    pt1 = (int(kpts[j1][0]), int(kpts[j1][1]))
                    pt2 = (int(kpts[j2][0]), int(kpts[j2][1]))
                    cv2.line(out_frame, pt1, pt2, (255, 0, 0), 2)
                    
        return out_frame
