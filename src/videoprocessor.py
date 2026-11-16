## video processing module

import ffmpeg

def load_crop_coordinates(path: str) -> dict:
	r"""Loads the crop coordinates from a specific file.

	Parameters
	----------
	path : str
		The full path to the coordinates file.

	Returns
	-------
	crop_coordinates : dict
		The dictionary containing the coordinates of the desired area to crop.

	Assumptions
	-------
		The file containing the crop coordinates needs to be in a specific
		format for the load to be succesful. Currently, the function loads
		.csv files in the format of "index,x1,x2,y1,y2".
	"""
	crop_coordinates = dict()
	crop_file = open(path, "r")
	crop_file_lines = crop_file.readlines()
	crop_file.close()
	variables = crop_file_lines[0].strip().split(",")[1:]

	for coord in [vf.strip().split(",") for vf in crop_file_lines[1:]]:
		crop_coordinates[coord[0]] = dict()
		for i, variable in enumerate(variables):
			crop_coordinates[coord[0]][variable] = int(coord[i+1])
	return crop_coordinates

def crop(path: str, coordinates: dict, output: str) -> None:
	r"""Crops a specific video file.

	Parameters
	----------
	path : str
		The full path to the input file.
	coordinates : dict
		A dictionary containing the cropping coordinates. The dictionary must
		have the fields "x", "y", "width" and "height", each with its respective
		values.
	output : str
		The full path to save the output file.

	Returns
	-------
	None

	"""
	(
	ffmpeg.input(
		path,
		err_detect='ignore_err'
	) 
	.crop(
		**coordinates
	)
	.output(
		str(output),
		vcodec='mpeg4',
		**{'qscale:v': 2}
	)
	.run()
	)

def trim(path: str, timestamps: dict, output: str) -> None:
	r"""Trims a specific video file.

	Parameters
	----------
	path : str
		The full path to the input file.
	timestamps : dict
		A dictionary containing the trimming timestamps. The dictionary must
		have the fields "ss" and "to", each with its respective values. The
		timestamps must be in "HH:MM:SS.mmm" format.
	output : str
		The full path to save the output file.

	Returns
	-------
	None

	"""
	(
	ffmpeg.input(
		path,
		**timestamps
	)
	.output(
		str(output),
		vcodec='copy',
		acodec='copy'
	)
	.run()
	)
	True

def read_video_info(path:str) -> int:
	"""
	Retrieve the height of a video in pixels
	Parameters
	----------
	path : str
		full path to a video file
	Returns
	-------
	height : int
		height of the video file in pixels
	"""
	probe = ffmpeg.probe(path)
	streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
	if len(streams) > 1:
		print('Videos with multiple streams are not supported. Analysing only first stream')

	video_info = streams[0]
	return video_info