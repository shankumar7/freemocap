import openvr
import math
import numpy as np
import time

class ViveTrackerClient:
    def __init__(self):
        self.vr_system = None
        self.is_initialized = False
        self.trackers = {}
        
    def initialize(self) -> bool:
        """Attempt to initialize OpenVR."""
        if self.is_initialized:
            return True
            
        try:
            # We use openvr.VRApplication_Other since we are not a VR game 
            # and just want to poll background tracking data.
            self.vr_system = openvr.init(openvr.VRApplication_Other)
            self.is_initialized = True
            print("Successfully initialized OpenVR.")
            return True
        except openvr.OpenVRError as e:
            print(f"Failed to initialize OpenVR: {e}")
            print("Make sure SteamVR is running and requireHmd is set to false in settings.")
            self.is_initialized = False
            return False

    def poll_poses(self) -> dict:
        """Poll the 6DoF poses for all connected devices."""
        if not self.is_initialized or not self.vr_system:
            return {}
            
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 
            0, # prediction seconds
            openvr.k_unMaxTrackedDeviceCount
        )
        
        tracker_data = {}
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            pose = poses[i]
            if pose.bPoseIsValid:
                device_class = self.vr_system.getTrackedDeviceClass(i)
                # We are interested in trackers, but let's grab controllers too just in case
                if device_class == openvr.TrackedDeviceClass_GenericTracker or device_class == openvr.TrackedDeviceClass_Controller:
                    
                    # Convert the 3x4 pose matrix into position and a 3x3 rotation matrix
                    matrix = pose.mDeviceToAbsoluteTracking
                    
                    x, y, z = matrix[0][3], matrix[1][3], matrix[2][3]
                    
                    # Rotation matrix
                    r_matrix = np.array([
                        [matrix[0][0], matrix[0][1], matrix[0][2]],
                        [matrix[1][0], matrix[1][1], matrix[1][2]],
                        [matrix[2][0], matrix[2][1], matrix[2][2]]
                    ])
                    
                    # Basic Euler angle extraction (yaw, pitch, roll) in radians
                    yaw = math.atan2(r_matrix[1, 0], r_matrix[0, 0])
                    pitch = math.atan2(-r_matrix[2, 0], math.sqrt(r_matrix[2, 1]**2 + r_matrix[2, 2]**2))
                    roll = math.atan2(r_matrix[2, 1], r_matrix[2, 2])
                    
                    # Retrieve the device serial number
                    serial = openvr.VRSystem().getStringTrackedDeviceProperty(
                        i, 
                        openvr.Prop_SerialNumber_String
                    )
                    
                    tracker_data[serial] = {
                        'position': (x, y, z),
                        'rotation_matrix': r_matrix,
                        'euler_angles': (yaw, pitch, roll),
                        'device_class': 'tracker' if device_class == openvr.TrackedDeviceClass_GenericTracker else 'controller',
                        'timestamp': time.time()
                    }
                    
        self.trackers = tracker_data
        return tracker_data
        
    def shutdown(self):
        if self.is_initialized:
            openvr.shutdown()
            self.is_initialized = False
            self.vr_system = None
            print("OpenVR shut down.")
