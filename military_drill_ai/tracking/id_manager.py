class IDManager:
    def __init__(self):
        self.next_id = 1
        self.yolo_to_cadet = {}

    def update_frame_tracks(self, current_yolo_ids):
        """
        If the frame is completely empty, reset the counter to 1.
        Otherwise, assign consecutive IDs to any new YOLO tracks.
        """
        if len(current_yolo_ids) == 0:
            self.next_id = 1
            self.yolo_to_cadet.clear()
            return

        for yolo_id in current_yolo_ids:
            if yolo_id not in self.yolo_to_cadet:
                self.yolo_to_cadet[yolo_id] = self.next_id
                self.next_id += 1

    def get_cadet_id(self, yolo_track_id):
        return self.yolo_to_cadet.get(yolo_track_id, -1)
