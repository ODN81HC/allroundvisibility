# All-round visibility test files
# This folder contains:
- The data collection tool, which is used to create subvideos from a large video, for the purpose of stimulating the camera views of a vehicle.
- Auto alignment tool, which is used to "re-construct" all-round view using the subvideos created as above, stimulate the ability to create a top-view of a vehicle.

# Data collection tool:
The file used: `dataSourceTool.py`.

How to use:
- In the file `test.ini`, specify the coordinates based on the template above, you can create more subvideos if you want using the same template. Moreover, specify the time, offset_x, offset_y if you want to stimulate the "collision/ dislocation" of the camera view. 
*Note*: The x_offset and y_offset values are with respect to the orginal coordinates, meaning that the next values in the list does not add up to the previous value, but they are just the offset rate with the original top-left coordinates.

-  Run the `dataSourceTool.py` by: python dataSourceTool.py
*Note*: Open the -h to get the parser help.

# Auto align tool:
The file used: `stitch_v2.py`

How to use:

*Note*: There is a specific feature in OpenCV which is the SURF feature extractors, which is deprecated in the newer version. Therefore, ***please downgrade the OpenCV version into <= 3.4.2 to use this feature***.

- Specify the `stitch_configs.py` parameters, and the explanation are there in the comment section.
- Specify the `--input` in the parser, which is the subvideos folder.
- **The subvideo files in the folder should be naming: `Left.mp4`, `Top.mp4`, `Right.mp4`, `Bottom.mp4`.**
- Run the tool by: python stitch_v2.py

# Other files (Optional to check it out)
`stitch_v1.py`: This is the 1st version of the image stitching using pure image feature extraction, matching and stitching. The result is not really good since the behaviour of the **perspectiveTransform** is not really robust. Hence, the outcome of the images are not good just because of that.

`oneToFourvids.py`: This is the first version of the data collection tool, which simply color the ractangles of the large video and see if the coordinates are being specified are correct or not.

`mdiApp.py`: A sketch based on Andy's work to get the four video running on four different threads.
