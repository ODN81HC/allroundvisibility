# This is the config file for the AutoAlignTool

# List of coordinates if frames are not matched
UNMATCHED_COORDS = [(0, 50), (50, 100), (100, 150), (150, 200)]
# List of output video names (must be in order)
# OUTPUT_VIDS = ['left.mp4', 'top.mp4', 'right.mp4', 'bottom.mp4']
OUTPUT_VIDS = ['-Y+Y.mp4', '+X-Y.mp4', '-Y-Y.mp4', '-X-Y.mp4']
# Left reference coordinate (if it matches with top frame)
LEFT_REF_COORDS = (200, 200)
# Background image size (test only)
# BACKGROUND_IMG = (1000, 1600, 3)
BACKGROUND_IMG = (3000, 3000, 3)

COMMISSION_TIME = 20
CHECK_INTERVAL = 10
MAX_UNCORR = 3

# Custom implementation
# Output videos to process the stitcher
OUTPUT_DIR = 'videos\\K360_output.mp4'
