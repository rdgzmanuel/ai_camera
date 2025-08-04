import numpy as np


class PredictiveROIProcessor:
    """
    Enhanced ROI processing that predicts where players will be based on movement patterns.
    """
    
    def __init__(self, frame_shape: tuple[int, int]):
        self.frame_h, self.frame_w = frame_shape
        self.player_velocities = {}  # Track player movement vectors
        self.prediction_horizon = 3  # Frames to predict ahead
        self.roi_expansion_factor = 1.4  # How much to expand predicted ROIs
        
    def update_player_velocities(self, tracks: list) -> None:
        """
        Update velocity estimates for tracked players.
        """
        for track in tracks:
            track_id = track.track_id
            current_pos = np.array([track.tlbr[0] + track.tlbr[2], 
                                   track.tlbr[1] + track.tlbr[3]]) / 2
            
            if track_id in self.player_velocities:
                prev_pos, prev_frame = self.player_velocities[track_id]
                velocity = current_pos - prev_pos
                self.player_velocities[track_id] = (current_pos, track.frame_id, velocity)
            else:
                self.player_velocities[track_id] = (current_pos, track.frame_id, np.array([0, 0]))
    
    def predict_player_positions(self, frames_ahead: int = 3) -> list[tuple[int, int, int, int]]:
        """
        Predict where players will be in future frames based on current velocity.
        """
        predicted_boxes = []
        
        for track_id, (pos, frame_id, velocity) in self.player_velocities.items():
            # Predict future position
            future_pos = pos + velocity * frames_ahead
            
            # Create bounding box around predicted position
            box_size = 60  # Estimated player size
            x1 = max(0, int(future_pos[0] - box_size))
            y1 = max(0, int(future_pos[1] - box_size))
            x2 = min(self.frame_w, int(future_pos[0] + box_size))
            y2 = min(self.frame_h, int(future_pos[1] + box_size))
            
            predicted_boxes.append((x1, y1, x2, y2))
        
        return predicted_boxes
    
    def cluster_boxes_to_rois(self, boxes: list[tuple[int, int, int, int]], 
                             max_rois: int) -> list[tuple[int, int, int, int]]:
        """
        Cluster boxes into ROIs using spatial proximity.
        """
        if not boxes:
            return []
        
        # Simple clustering based on center distances
        rois = []
        used_boxes = set()
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if i in used_boxes:
                continue
                
            # Find nearby boxes
            cluster_boxes = [(x1, y1, x2, y2)]
            used_boxes.add(i)
            
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            for j, (bx1, by1, bx2, by2) in enumerate(boxes):
                if j in used_boxes:
                    continue
                    
                b_center_x, b_center_y = (bx1 + bx2) // 2, (by1 + by2) // 2
                distance = np.sqrt((center_x - b_center_x)**2 + (center_y - b_center_y)**2)
                
                if distance < 200:  # Pixels
                    cluster_boxes.append((bx1, by1, bx2, by2))
                    used_boxes.add(j)
            
            # Create ROI from cluster
            if cluster_boxes:
                min_x = min(box[0] for box in cluster_boxes)
                min_y = min(box[1] for box in cluster_boxes)
                max_x = max(box[2] for box in cluster_boxes)
                max_y = max(box[3] for box in cluster_boxes)
                
                # Expand ROI
                width = max_x - min_x
                height = max_y - min_y
                expansion_w = int(width * (self.roi_expansion_factor - 1) / 2)
                expansion_h = int(height * (self.roi_expansion_factor - 1) / 2)
                
                roi_x1 = max(0, min_x - expansion_w)
                roi_y1 = max(0, min_y - expansion_h)
                roi_x2 = min(self.frame_w, max_x + expansion_w)
                roi_y2 = min(self.frame_h, max_y + expansion_h)
                
                rois.append((roi_x1, roi_y1, roi_x2, roi_y2))
                
                if len(rois) >= max_rois:
                    break
        
        return rois